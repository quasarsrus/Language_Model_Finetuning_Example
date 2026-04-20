from itertools import chain
from typing import Any

from datasets import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


class TinyShakespeareDataModule(LightningDataModule):
    """LightningDataModule` for the Tiny Shakespeare dataset.

    The Tiny Shakespeare dataset, compiled by Andrej Karpathy, consists of a single text file
    containing approximately 1 million characters or 40,000 lines of text derived from the
    complete works of William Shakespeare.

    This DataModule consists of two ways to obtain the dataset. The first way is a cleaner HuggingFace download
    from "https://huggingface.co/datasets/Trelis/tiny-shakespeare". This dataset is already split into train and test sets and is
    ready for tokenisation.

    The second way is by downloading it directly from
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt". The splits are currently performed
    as 80:10:10 (Train: Validation: Test), which can be adjusted in the self.setup() method. After the split, once the text is wrapped
    using HuggingFace's Dataset class, the rest of the process becomes the same as above.

    If an implementation is done using the first type of dataset, we will use the '<=====' symbol to identify it. Similarly,
    '===>' will be used for the second one. Commenting and uncommenting appropriately will yield the desired result.


    A `LightningDataModule` implements 7 key methods:

    ```pythonself
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.'''
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/smolLM-135M",
        data_dir: str = "data/",
        block_size: int = 128,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initialises the necessary components of the dataloader."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Prepares the dataset."""
        load_dataset(
            "Trelis/tiny-shakespeare", cache_dir=self.hparams.data_dir
        )  # <=====

        # os.makedirs(self.hparams.data_dir, exist_ok=True)                                                              # ===>
        # filepath = os.path.join(self.hparams.data_dir, "input.txt")                                                    # ===>
        # if not os.path.exists(filepath):                                                                               # ===>
        #    urllib.request.urlretrieve(                                                                                 # ===>
        #        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",            # ===>
        #       filepath                                                                                                 # ===>
        #            )                                                                                                   # ===>

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self.tokeniser = AutoTokenizer.from_pretrained(self.hparams.model_name)

        if self.tokeniser.pad_token is None:
            self.tokeniser.pad_token = self.tokeniser.eos_token

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokeniser, mlm=False
        )

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if self.data_train is None and self.data_val is None and self.data_test is None:
            dataset = load_dataset(
                "Trelis/tiny-shakespeare", cache_dir=self.hparams.data_dir
            )  # <=====

            split = dataset["train"].train_test_split(test_size=0.1, seed=35)  # <=====
            train = split["train"]  # <=====
            val = split["test"]  # <=====
            test = dataset["test"]  # <=====

            # filepath = os.path.join(self.hparams.data_dir, "input.txt")                                               # ===>
            # with open(filepath, "r") as f:                                                                            # ===>
            #    text = f.read()                                                                                        # ===>

            # n = len(text)                                                                                             # ===>
            # train_text = text[:int(n * 0.8)]                                                                          # ===>
            # val_text   = text[int(n * 0.8):int(n * 0.9)]                                                              # ===>
            # test_text  = text[int(n * 0.9):]                                                                          # ===>

            # train = Dataset.from_dict({"Text": [train_text]})                                                         # ===>
            # val   = Dataset.from_dict({"Text": [val_text]})                                                           # ===>
            # test  = Dataset.from_dict({"Text": [test_text]})                                                          # ===>

            def tokenise(example: dict) -> dict:
                return self.tokeniser(
                    example["Text"],
                    return_attention_mask=True,
                    truncation=False,
                    add_special_tokens=True,
                )

            def group_texts(examples: dict) -> dict:
                # concatenated = {k: sum(examples[k], []) for k in examples.keys()}
                concatenated = {
                    k: list(chain.from_iterable(examples[k])) for k in examples
                }
                total_length = len(concatenated["input_ids"])

                total_length = (
                    total_length // self.hparams.block_size
                ) * self.hparams.block_size

                result = {}

                for k, t in concatenated.items():
                    chunks = [
                        t[i : i + self.hparams.block_size]
                        for i in range(0, total_length, self.hparams.block_size)
                    ]
                    result[k] = chunks

                return result

                # chunks = []
                # for i in range(0, total_length, self.hparams.block_size):
                #    chunks.append(t[i : i + self.hparams.block_size])

            def process_split(split_dataset: Dataset) -> Dataset:
                tokenised = split_dataset.map(
                    tokenise, batched=True, remove_columns=["Text"]
                )
                return tokenised.map(group_texts, batched=True)

            self.data_train = process_split(train)
            self.data_val = process_split(val)
            self.data_test = process_split(test)

            self.data_train.set_format(type="torch")
            self.data_val.set_format(type="torch")
            self.data_test.set_format(type="torch")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.data_train is None:
            raise RuntimeError("data_train not initialized")

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.data_val is None:
            raise RuntimeError("data_train not initialized")

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.data_test is None:
            raise RuntimeError("data_train not initialized")

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collator,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up.

        Cleanup after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass


if __name__ == "__main__":
    _ = TinyShakespeareDataModule()

    """

    # Standalone snippet to check the module's correctness independently.'

    data_module = TinyShakespeareDataModule()
    data_module.prepare_data()
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    print(batch.keys())
    print(batch["input_ids"].shape)
    print(batch["labels"].shape)
    print(batch["input_ids"][0])
    print(batch["labels"][0])
    print(sum(batch["input_ids"][0] - batch["labels"][0]))
    """
