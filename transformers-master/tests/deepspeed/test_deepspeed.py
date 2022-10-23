# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import io
import json
import os
import unittest
from copy import deepcopy

from parameterized import parameterized
from transformers import TrainingArguments, is_torch_available
from transformers.file_utils import WEIGHTS_NAME
from transformers.integrations import is_deepspeed_available
from transformers.testing_utils import (
    CaptureLogger,
    ExtendSysPath,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    mockenv_context,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.trainer_utils import set_seed


bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa

    if is_torch_available():
        from test_trainer import get_regression_trainer  # noqa


set_seed(42)
MBART_TINY = "sshleifer/tiny-mbart"
T5_SMALL = "t5-small"


def load_json(path):
    with open(path) as f:
        return json.load(f)


# a candidate for testing_utils
def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    if not is_deepspeed_available():
        return unittest.skip("test requires deepspeed")(test_case)
    else:
        return test_case


ZERO2 = "zero2"
ZERO3 = "zero3"
stages = [ZERO2, ZERO3]


@require_deepspeed
@require_torch_gpu
class TrainerIntegrationDeepSpeed(TestCasePlus, TrainerIntegrationCommon):
    """

    This class is for testing directly via get_regression_trainer

    It mixes in `TrainerIntegrationCommon` which already has a lot of helper validation methods
    which we can re-use here.

    Important: this class' setup can only work with a single gpu because it runs within the current
    pytest worker. For multi-gpu tests use TestDeepSpeedWithLauncher.

    Note: if any of the tests of this class get run there will be at least one gpu occupied by them
    until this pytest worker exits. This is because the gpu memory allocated by the cuda-kernels
    won't be released until this pytest worker exits.

    This may appear as some run-away tests if you watch `nvidia-smi` while other tests that fork new
    processes are run. So there will be one or two "stale" processes reported in `nvidia-smi`. This
    is not a bug.
    """

    def setUp(self):
        super().setUp()

        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

        self.ds_config_file = {}
        self.ds_config_file[ZERO2] = f"{self.test_file_dir_str}/ds_config_zero2.json"
        self.ds_config_file[ZERO3] = f"{self.test_file_dir_str}/ds_config_zero3.json"

        # use self.get_config_dict(stage) to use these to ensure the original is not modified
        self.ds_config_dict = {}
        with io.open(self.ds_config_file[ZERO2], "r", encoding="utf-8") as f:
            self.ds_config_dict[ZERO2] = json.load(f)
        with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
            self.ds_config_dict[ZERO3] = json.load(f)

    def tearDown(self):
        # XXX: Fixme - this is a temporary band-aid since this global variable impacts other tests
        import transformers

        transformers.integrations._is_deepspeed_zero3_enabled = None

    def get_config_dict(self, stage):
        """ As the tests modify the dict, always make a copy """
        config = deepcopy(self.ds_config_dict[stage])
        if stage == ZERO3:
            # This setting slows things down, so don't enable it by default unless needed by a test.
            # It's in the file as a demo for users since we want everything to work out of the box even if slower.
            config["zero_optimization"]["stage3_gather_fp16_weights_on_model_save"] = False
        return config

    # --- These tests are enough to run on one of zero stages --- #

    # Test various combos
    # 1. DS scheduler + DS optimizer: this is already tested by most other tests
    # 2. HF scheduler + HF optimizer:
    # 3. DS scheduler + HF optimizer:
    # 4. HF scheduler + DS optimizer:

    def test_hf_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_ds_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_hf_scheduler_ds_optimizer(self):
        # this combo is not possible at the moment
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_zero2_dict)
            with self.assertRaises(Exception) as context:
                trainer.train()
        self.assertTrue("HF scheduler + DeepSpeed optimizer combination is not possible" in str(context.exception))

    def test_hf_optimizer_with_offload(self):
        # must not allow non-DS optimizer when using ZERO-offload
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = True
            # sanity check - should the default config change
            assert (
                "cpu_offload" in ds_config_zero2_dict["zero_optimization"]
                and ds_config_zero2_dict["zero_optimization"]["cpu_offload"] is True
            ), "ensure the config is set up correctly"
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_zero2_dict)
            with self.assertRaises(Exception) as context:
                trainer.train()
        self.assertTrue("ZeRO Offload can only work with DeepSpeed optimizers" in str(context.exception))

    # --- These tests need to run on both zero stages --- #
    @parameterized.expand(stages)
    def test_fake_notebook_no_launcher(self, stage):
        # this setup emulates a notebook where a launcher needs to be emulated by hand

        # note that unittest resets sys.stdout each test, so `CaptureStd` will work here to capture
        # DeepSpeed log if this test happens to run first in this pytest worker. But it will fail if
        # it's run not as a first test as `sys.stdout` will no longer be the same. So we either have
        # to reset `logger.handlers[0].setStream(sys.stdout)` or directly capture from the logger.
        from deepspeed.utils import logger

        with CaptureLogger(logger) as cs:
            with mockenv_context(**self.dist_env_1_gpu):
                trainer = get_regression_trainer(local_rank=0, deepspeed=self.ds_config_file[stage])
                trainer.train()
        assert "DeepSpeed info" in cs.out, "expected DeepSpeed logger output but got none"

    @parameterized.expand(stages)
    def test_early_get_last_lr(self, stage):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage,
        #
        # setting `logging_steps=1` forces an early `trainer._maybe_log_save_evaluate()` which calls
        # `self.lr_scheduler.get_last_lr()` and originally it'd fail on the very first step.
        with mockenv_context(**self.dist_env_1_gpu):
            a = b = 0.0
            trainer = get_regression_trainer(
                a=a,
                b=b,
                local_rank=0,
                train_len=8,
                deepspeed=self.ds_config_file[stage],
                per_device_train_batch_size=8,
                logging_steps=1,
            )
            trainer.train()
            post_train_a = trainer.model.a.item()

            # XXX: for some reason the following check fails with zero3 - not a broken but a
            # different qualitative outcome - need to investigate at some point
            if stage == ZERO3:
                return

            # it's enough that train didn't fail for this test, but we must check that
            # optimizer/scheduler didn't run (since if it did this test isn't testing the right thing)
            self.assertEqual(post_train_a, a)

    @parameterized.expand(stages)
    def test_gradient_accumulation(self, stage):
        # this test measures that we get identical weights and similar loss with:
        # 1. per_device_train_batch_size=8, gradient_accumulation_steps=1
        # 2. per_device_train_batch_size=4, gradient_accumulation_steps=2
        # since the 2nd should produce the effective batch of 1st, with the same results
        #
        # I can get an identical loss for a small train_len=32, plus the power of the initial
        # dynamic loss scale value set to:
        #   "fp16.initial_scale_power": 1
        # plus having the same WarmupLR's warmup_min_lr == warmup_max_lr in the config file
        # but for some reason going to train_len=64 the weights, weights start to mismatch with this setup.
        # the culprit seems to be `initial_scale_power` - putting it back to its default 32 keeps the weights identical

        train_len = 64
        a = b = 0.0

        with mockenv_context(**self.dist_env_1_gpu):
            no_grad_accum_trainer = get_regression_trainer(
                a=a,
                b=b,
                local_rank=0,
                train_len=train_len,
                deepspeed=self.ds_config_file[stage],
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
            )
            no_grad_accum_result = no_grad_accum_trainer.train()
            no_grad_accum_loss = no_grad_accum_result.training_loss
            no_grad_accum_a = no_grad_accum_trainer.model.a.item()
            no_grad_accum_b = no_grad_accum_trainer.model.b.item()
            # make sure the optimizer kicked in - if it hasn't changed from the original value of a then make train_len bigger
            self.assertNotEqual(no_grad_accum_a, a)

        with mockenv_context(**self.dist_env_1_gpu):
            yes_grad_accum_trainer = get_regression_trainer(
                a=a,
                b=b,
                local_rank=0,
                train_len=train_len,
                deepspeed=self.ds_config_file[stage],
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
            )
            yes_grad_accum_result = yes_grad_accum_trainer.train()
            yes_grad_accum_loss = yes_grad_accum_result.training_loss
            yes_grad_accum_a = yes_grad_accum_trainer.model.a.item()
            yes_grad_accum_b = yes_grad_accum_trainer.model.b.item()
            self.assertNotEqual(yes_grad_accum_a, a)

        # training with half the batch size but accumulation steps as 2 should give the same weights
        self.assertEqual(no_grad_accum_a, yes_grad_accum_a)
        self.assertEqual(no_grad_accum_b, yes_grad_accum_b)

        # see the note above how to get identical loss on a small bs
        self.assertAlmostEqual(no_grad_accum_loss, yes_grad_accum_loss, places=5)

    def check_saved_checkpoints_deepspeed(self, output_dir, freq, total, stage):
        # adapted from TrainerIntegrationCommon.check_saved_checkpoints

        file_list = [WEIGHTS_NAME, "training_args.bin", "trainer_state.json", "config.json"]

        if stage == ZERO2:
            ds_file_list = ["mp_rank_00_model_states.pt"]
        elif stage == ZERO3:
            ds_file_list = ["zero_pp_rank_0_mp_rank_00_model_states.pt"]
        else:
            raise ValueError(f"unknown stage {stage}")

        # XXX: this can be recoded and then removed once we require deepspeed>0.3.13
        from packaging import version

        import deepspeed

        if version.parse(deepspeed.__version__) > version.parse("0.3.13"):
            ds_file_list.append("zero_pp_rank_0_mp_rank_00_optim_states.pt")
        else:
            ds_file_list.append("zero_pp_rank_0_mp_rank_00optim_states.pt")

        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint), f"[{stage}] {checkpoint} dir is not found")

            # common files
            for filename in file_list:
                path = os.path.join(checkpoint, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

            # ds files
            ds_path = os.path.join(checkpoint, f"global_step{step}")
            for filename in ds_file_list:
                # filename = os.path.join(path, filename)
                # print(filename)
                path = os.path.join(ds_path, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

    @parameterized.expand(stages)
    def test_save_checkpoints(self, stage):
        # adapted from  TrainerIntegrationTest.test_save_checkpoints

        freq = 5
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_fp16_weights_on_model_save"] = True

        # save checkpoints
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                output_dir=output_dir,
                save_steps=freq,
                deepspeed=ds_config_dict,
            )
            trainer.train()

        total = int(self.n_epochs * 64 / self.batch_size)
        self.check_saved_checkpoints_deepspeed(output_dir, freq, total, stage)

    @parameterized.expand(stages)
    def test_can_resume_training_errors(self, stage):

        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = self.get_config_dict(stage)
            output_dir = self.get_auto_remove_tmp_dir()
            trainer = get_regression_trainer(output_dir=output_dir, deepspeed=ds_config_dict)

            # 1. fail to find any checkpoint - due a fresh output_dir
            with self.assertRaises(Exception) as context:
                trainer.train(resume_from_checkpoint=True)
            self.assertTrue(
                "No valid checkpoint found in output directory" in str(context.exception),
                f"got exception: {context.exception}",
            )

            # 2. fail to find a bogus checkpoint
            with self.assertRaises(Exception) as context:
                checkpoint = os.path.join(output_dir, "checkpoint-5")
                trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
            self.assertTrue(
                "Can't find a valid checkpoint at" in str(context.exception), f"got exception: {context.exception}"
            )

    @parameterized.expand(stages)
    def test_can_resume_training_normal(self, stage):
        # adapted from TrainerIntegrationTest.test_can_resume_training
        # test normal resume for each stage separately, error-handling is tested in a different test
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_fp16_weights_on_model_save"] = True

        kwargs = dict(output_dir=output_dir, train_len=128, save_steps=5, learning_rate=0.1, deepspeed=ds_config_dict)

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(output_dir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(output_dir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)


@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeedWithLauncher(TestCasePlus):
    """ This class is for testing via an external script - can do multiple gpus """

    # Tests to devise #
    #
    # 1. predict_with_generate on multigpu - need to figure out how to give input sequences so that
    # the 2 gpus will generate prediction sequences that aren't of the same length - this is because
    # we had to code a special feature to sync the gpus when the predicted sequences aren't of the
    # same length. In general this will tested as a side-effect through a variety of other tests -
    # it'll simply hang trying to synchronize with other gpus if this problem is encountered. So as
    # long as we have a few full tests running on zero3 + predict_with_generate this should be
    # mostly covered.
    #
    # but there are 5 variations on beam search in `generate`- with identical code branched with `if
    # synced_gpus`
    #
    # 2. most tests should probably be run on both: zero2 and zero3 configs
    #

    @require_torch_multi_gpu
    @parameterized.expand(stages)
    def test_basic_distributed(self, stage):
        self.run_and_check(stage=stage, distributed=True)

    @parameterized.expand(stages)
    def test_do_eval_no_train(self, stage):
        # we should not fail if train is skipped
        self.run_and_check(
            stage=stage,
            eval_steps=1,
            distributed=False,
            do_train=False,
            do_eval=True,
        )

    @parameterized.expand(stages)
    def test_resume_train_not_from_ds_checkpoint(self, stage):
        # do normal training and then resume not from the deepspeed checkpoint but explicitly from
        # the saved model dir

        do_train = True
        do_eval = False
        kwargs = dict(stage=stage, eval_steps=1, distributed=True, do_train=do_train, do_eval=do_eval)

        # 1. normal training
        output_dir = self.run_and_check(**kwargs)

        # 2. now resume explicitly from the saved weights, by passing --model_name_or_path output_dir
        # - i.e. the same path the model was saved to in step 1
        output_dir = self.run_trainer(**kwargs, model_name=output_dir)

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval)

    def do_checks(self, output_dir, do_train=True, do_eval=True):

        if do_train:
            train_metrics = load_json(os.path.join(output_dir, "train_results.json"))
            self.assertIn("train_samples_per_second", train_metrics)
            self.assertGreater(train_metrics["train_samples_per_second"], 0.5)

        if do_eval:
            eval_metrics = load_json(os.path.join(output_dir, "eval_results.json"))
            self.assertIn("eval_bleu", eval_metrics)
            self.assertGreater(eval_metrics["eval_bleu"], 0)

    # XXX: need to do better validation beyond just that the run was successful
    def run_and_check(
        self,
        stage,
        eval_steps=10,
        distributed=True,
        do_train=True,
        do_eval=True,
        extra_args_str=None,
        remove_args_str=None,
    ):

        # we are doing quality testing so using a small real model
        output_dir = self.run_trainer(
            stage=stage,
            model_name=T5_SMALL,
            eval_steps=eval_steps,
            num_train_epochs=1,
            do_train=do_train,
            do_eval=do_eval,
            distributed=distributed,
            extra_args_str=extra_args_str,
            remove_args_str=remove_args_str,
        )

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval)

        return output_dir

    def run_trainer(
        self,
        stage: str,
        model_name: str,
        eval_steps: int = 10,
        num_train_epochs: int = 1,
        do_train: bool = False,
        do_eval: bool = True,
        distributed: bool = True,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):
        max_len = 32
        data_dir = self.test_file_dir / "../fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {model_name}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --max_source_length {max_len}
            --max_target_length {max_len}
            --val_max_target_length {max_len}
            --warmup_steps 8
            --predict_with_generate
            --logging_steps 0
            --save_steps 0
            --eval_steps {eval_steps}
            --group_by_length
            --label_smoothing_factor 0.1
            --adafactor
            --source_lang en
            --target_lang ro
        """.split()
        args.extend(["--source_prefix", '"translate English to Romanian: "'])

        actions = 0
        if do_train:
            actions += 1
            args.extend(
                f"""
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --max_train_samples 100
            --per_device_train_batch_size 2
            --learning_rate 3e-3
            """.split()
            )

        if do_eval:
            actions += 1
            args.extend(
                """
            --do_eval
            --max_val_samples 100
            --per_device_eval_batch_size 2
            """.split()
            )

        assert actions > 0, "need at least do_train or do_eval for the test to run"

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        # currently only works for bool args
        if remove_args_str is not None:
            remove_args = remove_args_str.split()
            args = [x for x in args if x not in remove_args]

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/translation/run_translation.py"]
        launcher = self.get_launcher(distributed)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir

    @parameterized.expand(stages)
    def test_clm(self, stage):
        # this test exercises model.resize_token_embeddings() which requires param gathering outside
        # of forward - it's not used by `run_translation.py`, but it is in `run_clm.py`

        data_dir = self.tests_dir / "fixtures"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path sshleifer/tiny-gpt2
            --train_file {data_dir}/sample_text.txt
            --validation_file {data_dir}/sample_text.txt
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --max_train_samples 10
            --max_val_samples 10
            --per_device_train_batch_size 5
            --per_device_eval_batch_size 5
            --num_train_epochs 1
            --warmup_steps 8
            --block_size 128
            """.split()

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/language-modeling/run_clm.py"]
        launcher = self.get_launcher(distributed=True)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir

    def get_launcher(self, distributed=False):
        # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
        # - it won't be able to handle that
        # 2. for now testing with just 2 gpus max (since some quality tests may give different
        # results with mode gpus because we use very little data)
        num_gpus = min(2, get_gpu_count()) if distributed else 1
        return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()
