from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast


class TinyShakespeareModule(LightningModule):
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/smolLM-135M",
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        torch_compile: bool = False,
        prompt_fraction: float = 0.5,
        max_new_tokens: int | None = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # the Low Rank Adaptation method works well when data is limited and model could
        # easily overfit.
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
            )
            self.net = get_peft_model(base_model, lora_config)
            self.net.print_trainable_parameters()
        else:
            self.net = base_model

        self.tokenizer: PreTrainedTokenizerBase | None = None

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        self.test_rouge = ROUGEScore(rouge_keys="rougeL")

    def forward(
        self, input_ids, attention_mask=None, labels=None
    ) -> CausalLMOutputWithPast:
        """Perform a forward pass through the model `self.net`.

        :param input_ids: Token ids of shape (batch_size, block_size)
        :param attention_mask: Mask of shape (batch_size, block_size)
        :param labels: Token ids for loss computation,

        :return: CausalLMOutputWithPast containing loss and logits.
        """
        return self.net(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits

        return loss, logits

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        self.log(
            "train/perplexity", torch.exp(self.train_loss.compute()), prog_bar=True
        )
        self.train_loss.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        self.val_loss_best(val_loss)

        self.log("val/perplexity", torch.exp(val_loss), prog_bar=True)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "val/perplexity_best",
            torch.exp(self.val_loss_best.compute()),
            prog_bar=True,
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        assert self.tokenizer is not None
        loss, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_len = int(input_ids.shape[1] * self.hparams.prompt_fraction)

        prompts = input_ids[:, :prompt_len]
        prompt_mask = attention_mask[:, :prompt_len]
        references = input_ids[:, prompt_len:]

        with torch.no_grad():
            generated = self.net.generate(
                input_ids=prompts,
                attention_mask=prompt_mask,
                max_new_tokens=self.hparams.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_continuation = generated[:, prompt_len:]

        preds = self.tokenizer.batch_decode(
            generated_continuation, skip_special_tokens=True
        )
        refs = self.tokenizer.batch_decode(references, skip_special_tokens=True)

        self.test_rouge(preds, refs)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/perplexity", torch.exp(self.test_loss.compute()), prog_bar=True)
        self.test_loss.reset()

        rouge_scores = self.test_rouge.compute()
        self.log("test/rougeL_fmeasure", rouge_scores["rougeL_fmeasure"], prog_bar=True)
        self.log(
            "test/rougeL_precision", rouge_scores["rougeL_precision"], prog_bar=True
        )
        self.log("test/rougeL_recall", rouge_scores["rougeL_recall"], prog_bar=True)
        self.test_rouge.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.hparams.torch_compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.

        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    pass
