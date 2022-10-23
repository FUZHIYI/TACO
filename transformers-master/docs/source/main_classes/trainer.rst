..
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Trainer
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.Trainer` and :class:`~transformers.TFTrainer` classes provide an API for feature-complete
training in most standard use cases. It's used in most of the :doc:`example scripts <../examples>`.

Before instantiating your :class:`~transformers.Trainer`/:class:`~transformers.TFTrainer`, create a
:class:`~transformers.TrainingArguments`/:class:`~transformers.TFTrainingArguments` to access all the points of
customization during training.

The API supports distributed training on multiple GPUs/TPUs, mixed precision through `NVIDIA Apex
<https://github.com/NVIDIA/apex>`__ and Native AMP for PyTorch and :obj:`tf.keras.mixed_precision` for TensorFlow.

Both :class:`~transformers.Trainer` and :class:`~transformers.TFTrainer` contain the basic training loop which supports
the above features. To inject custom behavior you can subclass them and override the following methods:

- **get_train_dataloader**/**get_train_tfdataset** -- Creates the training DataLoader (PyTorch) or TF Dataset.
- **get_eval_dataloader**/**get_eval_tfdataset** -- Creates the evaluation DataLoader (PyTorch) or TF Dataset.
- **get_test_dataloader**/**get_test_tfdataset** -- Creates the test DataLoader (PyTorch) or TF Dataset.
- **log** -- Logs information on the various objects watching training.
- **create_optimizer_and_scheduler** -- Sets up the optimizer and learning rate scheduler if they were not passed at
  init. Note, that you can also subclass or override the ``create_optimizer`` and ``create_scheduler`` methods
  separately.
- **create_optimizer** -- Sets up the optimizer if it wasn't passed at init.
- **create_scheduler** -- Sets up the learning rate scheduler if it wasn't passed at init.
- **compute_loss** - Computes the loss on a batch of training inputs.
- **training_step** -- Performs a training step.
- **prediction_step** -- Performs an evaluation/test step.
- **run_model** (TensorFlow only) -- Basic pass through the model.
- **evaluate** -- Runs an evaluation loop and returns metrics.
- **predict** -- Returns predictions (with metrics if labels are available) on a test set.

.. warning::

    The :class:`~transformers.Trainer` class is optimized for 🤗 Transformers models and can have surprising behaviors
    when you use it on other models. When using it on your own model, make sure:

    - your model always return tuples or subclasses of :class:`~transformers.file_utils.ModelOutput`.
    - your model can compute the loss if a :obj:`labels` argument is provided and that loss is returned as the first
      element of the tuple (if your model returns tuples)
    - your model can accept multiple label arguments (use the :obj:`label_names` in your
      :class:`~transformers.TrainingArguments` to indicate their name to the :class:`~transformers.Trainer`) but none
      of them should be named :obj:`"label"`.

Here is an example of how to customize :class:`~transformers.Trainer` using a custom loss function for multi-label
classification:

.. code-block:: python

    import torch
    from transformers import Trainer

    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.float().view(-1, self.model.config.num_labels))
            return (loss, outputs) if return_outputs else loss

Another way to customize the training loop behavior for the PyTorch :class:`~transformers.Trainer` is to use
:doc:`callbacks <callback>` that can inspect the training loop state (for progress reporting, logging on TensorBoard or
other ML platforms...) and take decisions (like early stopping).


Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Trainer
    :members:


Seq2SeqTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Seq2SeqTrainer
    :members: evaluate, predict


TFTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTrainer
    :members:


TrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TrainingArguments
    :members:


Seq2SeqTrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Seq2SeqTrainingArguments
    :members:


TFTrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTrainingArguments
    :members:


Trainer Integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The :class:`~transformers.Trainer` has been extended to support libraries that may dramatically improve your training
time and fit much bigger models.

Currently it supports third party solutions, `DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ and `FairScale
<https://github.com/facebookresearch/fairscale/>`__, which implement parts of the paper `ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models, by Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
<https://arxiv.org/abs/1910.02054>`__.

This provided support is new and experimental as of this writing.

.. _zero-install-notes:

Installation Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of this writing, both FairScale and Deepspeed require compilation of CUDA C++ code, before they can be used.

While all installation issues should be dealt with through the corresponding GitHub Issues of `FairScale
<https://github.com/facebookresearch/fairscale/issues>`__ and `Deepspeed
<https://github.com/microsoft/DeepSpeed/issues>`__, there are a few common issues that one may encounter while building
any PyTorch extension that needs to build CUDA extensions.

Therefore, if you encounter a CUDA-related build issue while doing one of the following or both:

.. code-block:: bash

    pip install fairscale
    pip install deepspeed

please, read the following notes first.

In these notes we give examples for what to do when ``pytorch`` has been built with CUDA ``10.2``. If your situation is
different remember to adjust the version number to the one you are after.

Possible problem #1
=======================================================================================================================

While, Pytorch comes with its own CUDA toolkit, to build these two projects you must have an identical version of CUDA
installed system-wide.

For example, if you installed ``pytorch`` with ``cudatoolkit==10.2`` in the Python environment, you also need to have
CUDA ``10.2`` installed system-wide.

The exact location may vary from system to system, but ``/usr/local/cuda-10.2`` is the most common location on many
Unix systems. When CUDA is correctly set up and added to the ``PATH`` environment variable, one can find the
installation location by doing:

.. code-block:: bash

    which nvcc

If you don't have CUDA installed system-wide, install it first. You will find the instructions by using your favorite
search engine. For example, if you're on Ubuntu you may want to search for: `ubuntu cuda 10.2 install
<https://www.google.com/search?q=ubuntu+cuda+10.2+install>`__.

Possible problem #2
=======================================================================================================================

Another possible common problem is that you may have more than one CUDA toolkit installed system-wide. For example you
may have:

.. code-block:: bash

    /usr/local/cuda-10.2
    /usr/local/cuda-11.0

Now, in this situation you need to make sure that your ``PATH`` and ``LD_LIBRARY_PATH`` environment variables contain
the correct paths to the desired CUDA version. Typically, package installers will set these to contain whatever the
last version was installed. If you encounter the problem, where the package build fails because it can't find the right
CUDA version despite you having it installed system-wide, it means that you need to adjust the 2 aforementioned
environment variables.

First, you may look at their contents:

.. code-block:: bash

    echo $PATH
    echo $LD_LIBRARY_PATH

so you get an idea of what is inside.

It's possible that ``LD_LIBRARY_PATH`` is empty.

``PATH`` lists the locations of where executables can be found and ``LD_LIBRARY_PATH`` is for where shared libraries
are to looked for. In both cases, earlier entries have priority over the later ones. ``:`` is used to separate multiple
entries.

Now, to tell the build program where to find the specific CUDA toolkit, insert the desired paths to be listed first by
doing:

.. code-block:: bash

    export PATH=/usr/local/cuda-10.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

Note that we aren't overwriting the existing values, but prepending instead.

Of course, adjust the version number, the full path if need be. Check that the directories you assign actually do
exist. ``lib64`` sub-directory is where the various CUDA ``.so`` objects, like ``libcudart.so`` reside, it's unlikely
that your system will have it named differently, but if it is adjust it to reflect your reality.


Possible problem #3
=======================================================================================================================

Some older CUDA versions may refuse to build with newer compilers. For example, you my have ``gcc-9`` but it wants
``gcc-7``.

There are various ways to go about it.

If you can install the latest CUDA toolkit it typically should support the newer compiler.

Alternatively, you could install the lower version of the compiler in addition to the one you already have, or you may
already have it but it's not the default one, so the build system can't see it. If you have ``gcc-7`` installed but the
build system complains it can't find it, the following might do the trick:

.. code-block:: bash

    sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
    sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++


Here, we are making a symlink to ``gcc-7`` from ``/usr/local/cuda-10.2/bin/gcc`` and since
``/usr/local/cuda-10.2/bin/`` should be in the ``PATH`` environment variable (see the previous problem's solution), it
should find ``gcc-7`` (and ``g++7``) and then the build will succeed.

As always make sure to edit the paths in the example to match your situation.

FairScale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By integrating `FairScale <https://github.com/facebookresearch/fairscale/>`__ the :class:`~transformers.Trainer`
provides support for the following features from `the ZeRO paper <https://arxiv.org/abs/1910.02054>`__:

1. Optimizer State Sharding
2. Gradient Sharding
3. Model Parameters Sharding (new and very experimental)
4. CPU offload (new and very experimental)

You will need at least two GPUs to use this feature.


**Installation**:

Install the library via pypi:

.. code-block:: bash

    pip install fairscale

or via ``transformers``' ``extras``:

.. code-block:: bash

    pip install transformers[fairscale]

(will become available starting from ``transformers==4.6.0``)

or find more details on `the FairScale's GitHub page <https://github.com/facebookresearch/fairscale/#installation>`__.

If you're still struggling with the build, first make sure to read :ref:`zero-install-notes`.

If it's still not resolved the build issue, here are a few more ideas.

``fairscale`` seems to have an issue with the recently introduced by pip build isolation feature. If you have a problem
with it, you may want to try one of:

.. code-block:: bash

    pip install fairscale --no-build-isolation .

or:

.. code-block:: bash

    git clone https://github.com/facebookresearch/fairscale/
    cd fairscale
    rm -r dist build
    python setup.py bdist_wheel
    pip uninstall -y fairscale
    pip install dist/fairscale-*.whl

``fairscale`` also has issues with building against pytorch-nightly, so if you use it you may have to try one of:

.. code-block:: bash

    pip uninstall -y fairscale; pip install fairscale --pre \
    -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html \
    --no-cache --no-build-isolation

or:

.. code-block:: bash

    pip install -v --disable-pip-version-check . \
    -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html --pre

Of course, adjust the urls to match the cuda version you use.

If after trying everything suggested you still encounter build issues, please, proceed with the GitHub Issue of
`FairScale <https://github.com/facebookresearch/fairscale/issues>`__.



**Usage**:

To use the first version of Sharded data-parallelism, add ``--sharded_ddp simple`` to the command line arguments, and
make sure you have added the distributed launcher ``-m torch.distributed.launch
--nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`` if you haven't been using it already.

For example here is how you could use it for ``run_translation.py`` with 2 GPUs:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro \
    --fp16 --sharded_ddp simple

Notes:

- This feature requires distributed training (so multiple GPUs).
- It is not implemented for TPUs.
- It works with ``--fp16`` too, to make things even faster.
- One of the main benefits of enabling ``--sharded_ddp simple`` is that it uses a lot less GPU memory, so you should be
  able to use significantly larger batch sizes using the same hardware (e.g. 3x and even bigger) which should lead to
  significantly shorter training time.

3. To use the second version of Sharded data-parallelism, add ``--sharded_ddp zero_dp_2`` or ``--sharded_ddp
   zero_dp_3`` to the command line arguments, and make sure you have added the distributed launcher ``-m
   torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`` if you haven't been using it already.

For example here is how you could use it for ``run_translation.py`` with 2 GPUs:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro \
    --fp16 --sharded_ddp zero_dp_2

:obj:`zero_dp_2` is an optimized version of the simple wrapper, while :obj:`zero_dp_3` fully shards model weights,
gradients and optimizer states.

Both are compatible with adding :obj:`cpu_offload` to enable ZeRO-offload (activate it like this: :obj:`--sharded_ddp
"zero_dp_2 cpu_offload"`).

Notes:

- This feature requires distributed training (so multiple GPUs).
- It is not implemented for TPUs.
- It works with ``--fp16`` too, to make things even faster.
- The ``cpu_offload`` additional option requires ``--fp16``.
- This is an area of active development, so make sure you have a source install of fairscale to use this feature as
  some bugs you encounter may have been fixed there already.

Known caveats:

- This feature is incompatible with :obj:`--predict_with_generate` in the `run_translation.py` script.
- Using :obj:`--sharded_ddp zero_dp_3` requires wrapping each layer of the model in the special container
  :obj:`FullyShardedDataParallelism` of fairscale. It should be used with the option :obj:`auto_wrap` if you are not
  doing this yourself: :obj:`--sharded_ddp "zero_dp_3 auto_wrap"`.


DeepSpeed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ implements everything described in the `ZeRO paper
<https://arxiv.org/abs/1910.02054>`__. Currently it provides full support for:

1. Optimizer State Partitioning (ZeRO stage 1)
2. Gradient Partitioning (ZeRO stage 2)
3. Param Partitioning (ZeRO stage 3)
4. Custom mixed precision training handling
5. A range of fast CUDA-extension-based Optimizers
6. ZeRO-Offload

ZeRO-Offload has its own dedicated paper: `ZeRO-Offload: Democratizing Billion-Scale Model Training
<https://arxiv.org/abs/2101.06840>`__.

DeepSpeed ZeRO-2 is currently used only for training, as all the currently available features are of no use to
inference.

DeepSpeed ZeRO-3 can be used for inference as well, since it allows huge models to be loaded on multiple GPUs, which
won't be possible on a single GPU.



Installation
=======================================================================================================================

Install the library via pypi:

.. code-block:: bash

    pip install deepspeed

or via ``transformers``' ``extras``:

.. code-block:: bash

    pip install transformers[deepspeed]

(will become available starting from ``transformers==4.6.0``)

or find more details on `the DeepSpeed's GitHub page <https://github.com/microsoft/deepspeed#installation>`__ and
`advanced install <https://www.deepspeed.ai/tutorials/advanced-install/>`__.

If you're still struggling with the build, first make sure to read :ref:`zero-install-notes`.

If you don't prebuild the extensions and rely on them to be built at run time and you tried all of the above solutions
to no avail, the next thing to try is to pre-build the modules before installing them.

To make a local build for DeepSpeed:

.. code-block:: bash

    git clone https://github.com/microsoft/DeepSpeed/
    cd DeepSpeed
    rm -rf build
    TORCH_CUDA_ARCH_LIST="6.1;8.6" DS_BUILD_OPS=1 pip install . \
    --global-option="build_ext" --global-option="-j8" --no-cache -v \
    --disable-pip-version-check 2>&1 | tee build.log

Edit ``TORCH_CUDA_ARCH_LIST`` to insert the code for the architectures of the GPU cards you intend to use.

Or if you need to use the same setup on multiple machines, make a binary wheel:

.. code-block:: bash

    git clone https://github.com/microsoft/DeepSpeed/
    cd DeepSpeed
    rm -rf build
    TORCH_CUDA_ARCH_LIST="6.1;8.6" DS_BUILD_OPS=1 \
    python setup.py build_ext -j8 bdist_wheel

it will generate something like ``dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`` which now you can install
as ``pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`` locally or on any other machine.

Again, remember to ensure to adjust ``TORCH_CUDA_ARCH_LIST`` to the target architectures.

You can find the complete list of NVIDIA GPUs and their corresponding **Compute Capabilities** (same as arch in this
context) `here <https://developer.nvidia.com/cuda-gpus>`__.

You can check the archs pytorch was built with using:

.. code-block:: bash

    python -c "import torch; print(torch.cuda.get_arch_list())"

Here is how to find out the arch for one of the installed GPU. For example, for GPU 0:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
    print(torch.cuda.get_device_properties(torch.device('cuda')))"

If the output is:

.. code-block:: bash

    _CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)

then you know that this card's arch is ``8.6``.

You can also leave ``TORCH_CUDA_ARCH_LIST`` out completely and then the build program will automatically query the
architecture of the GPUs the build is made on. This may or may not match the GPUs on the target machines, that's why
it's best to specify the desired archs explicitly.

If after trying everything suggested you still encounter build issues, please, proceed with the GitHub Issue of
`Deepspeed <https://github.com/microsoft/DeepSpeed/issues>`__,



Deployment with multiple GPUs
=======================================================================================================================

To deploy this feature with multiple GPUs adjust the :class:`~transformers.Trainer` command line arguments as
following:

1. replace ``python -m torch.distributed.launch`` with ``deepspeed``.
2. add a new argument ``--deepspeed ds_config.json``, where ``ds_config.json`` is the DeepSpeed configuration file as
   documented `here <https://www.deepspeed.ai/docs/config-json/>`__. The file naming is up to you.

Therefore, if your original command line looked as following:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 your_program.py <normal cl args>

Now it should be:

.. code-block:: bash

    deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json

Unlike, ``torch.distributed.launch`` where you have to specify how many GPUs to use with ``--nproc_per_node``, with the
``deepspeed`` launcher you don't have to use the corresponding ``--num_gpus`` if you want all of your GPUs used. The
full details on how to configure various nodes and GPUs can be found `here
<https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node>`__.

In fact, you can continue using ``-m torch.distributed.launch`` with DeepSpeed as long as you don't need to use
``deepspeed`` launcher-specific arguments. Typically if you don't need a multi-node setup you're not required to use
the ``deepspeed`` launcher. But since in the DeepSpeed documentation it'll be used everywhere, for consistency we will
use it here as well.

Here is an example of running ``run_translation.py`` under DeepSpeed deploying all available GPUs:

.. code-block:: bash

    deepspeed examples/pytorch/translation/run_translation.py \
    --deepspeed tests/deepspeed/ds_config.json \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir --fp16 \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro


Note that in the DeepSpeed documentation you are likely to see ``--deepspeed --deepspeed_config ds_config.json`` - i.e.
two DeepSpeed-related arguments, but for the sake of simplicity, and since there are already so many arguments to deal
with, we combined the two into a single argument.

For some practical usage examples, please, see this `post
<https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400>`__.



Deployment with one GPU
=======================================================================================================================

To deploy DeepSpeed with one GPU adjust the :class:`~transformers.Trainer` command line arguments as following:

.. code-block:: bash

    deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
    --deepspeed tests/deepspeed/ds_config.json \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir --fp16 \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro

This is almost the same as with multiple-GPUs, but here we tell DeepSpeed explicitly to use just one GPU. By default,
DeepSpeed deploys all GPUs it can see. If you have only 1 GPU to start with, then you don't need this argument. The
following `documentation <https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node>`__ discusses the
launcher options.

Why would you want to use DeepSpeed with just one GPU?

1. It has a ZeRO-offload feature which can delegate some computations and memory to the host's CPU and RAM, and thus
   leave more GPU resources for model's needs - e.g. larger batch size, or enabling a fitting of a very big model which
   normally won't fit.
2. It provides a smart GPU memory management system, that minimizes memory fragmentation, which again allows you to fit
   bigger models and data batches.

While we are going to discuss the configuration in details next, the key to getting a huge improvement on a single GPU
with DeepSpeed is to have at least the following configuration in the configuration file:

.. code-block:: json

    {
      "zero_optimization": {
         "stage": 2,
         "allgather_partitions": true,
         "allgather_bucket_size": 2e8,
         "reduce_scatter": true,
         "reduce_bucket_size": 2e8,
         "overlap_comm": true,
         "contiguous_gradients": true,
         "cpu_offload": true
      },
    }

which enables ``cpu_offload`` and some other important features. You may experiment with the buffer sizes, you will
find more details in the discussion below.

For a practical usage example of this type of deployment, please, see this `post
<https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685>`__.

Notes:

- if you need to run on a specific GPU, which is different from GPU 0, you can't use ``CUDA_VISIBLE_DEVICES`` to limit
  the visible scope of available GPUs. Instead, you have to use the following syntax:

   .. code-block:: bash

       deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...

   In this example, we tell DeepSpeed to use GPU 1 (second gpu).



Deployment in Notebooks
=======================================================================================================================

The problem with running notebook cells as a script is that there is no normal ``deepspeed`` launcher to rely on, so
under certain setups we have to emulate it.

If you're using only 1 GPU, here is how you'd have to adjust your training code in the notebook to use DeepSpeed.

.. code-block:: python

    # DeepSpeed requires a distributed environment even when only one process is used.
    # This emulates a launcher in the notebook
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"

    # Now proceed as normal, plus pass the deepspeed config file
    training_args = TrainingArguments(..., deepspeed="ds_config.json")
    trainer = Trainer(...)
    trainer.train()

Note: ``...`` stands for the normal arguments that you'd pass to the functions.

If you want to use more than 1 GPU, you must use a multi-process environment for DeepSpeed to work. That is, you have
to use the launcher for that purpose and this cannot be accomplished by emulating the distributed environment presented
at the beginning of this section.

If you want to create the config file on the fly in the notebook in the current directory, you could have a dedicated
cell with:

.. code-block:: python

    %%bash
    cat <<'EOT' > ds_config.json
    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": true,
            "allgather_bucket_size": 2e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true,
            "cpu_offload": true
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500
            }
        },

        "steps_per_print": 2000,
        "wall_clock_breakdown": false
    }
    EOT


If the training script is in a normal file and not in the notebook cells, you can launch ``deepspeed`` normally via
shell from a cell. For example, to use ``run_translation.py`` you would launch it with:

.. code-block::

    !git clone https://github.com/huggingface/transformers
    !cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...

or with ``%%bash`` magic, where you can write a multi-line code for the shell program to run:

.. code-block::

    %%bash

    git clone https://github.com/huggingface/transformers
    cd transformers
    deepspeed examples/pytorch/translation/run_translation.py ...

In such case you don't need any of the code presented at the beginning of this section.

Note: ``%%bash`` magic is neat, but currently it buffers the output so you won't see the logs until the process
completes.





Configuration
=======================================================================================================================

For the complete guide to the DeepSpeed configuration options that can be used in its configuration file please refer
to the `following documentation <https://www.deepspeed.ai/docs/config-json/>`__.

You can find dozens of DeepSpeed configuration examples that address various practical needs in `the DeepSpeedExamples
repo <https://github.com/microsoft/DeepSpeedExamples>`__:

.. code-block:: bash

    git clone https://github.com/microsoft/DeepSpeedExamples
    cd DeepSpeedExamples
    find . -name '*json'

Continuing the code from above, let's say you're looking to configure the Lamb optimizer. So you can search through the
example ``.json`` files with:

.. code-block:: bash

    grep -i Lamb $(find . -name '*json')

Some more examples are to be found in the `main repo <https://github.com/microsoft/DeepSpeed>`__ as well.

When using DeepSpeed you always need to supply a DeepSpeed configuration file, yet some configuration parameters have
to be configured via the command line. You will find the nuances in the rest of this guide.

To get an idea of what DeepSpeed configuration file looks like, here is one that activates ZeRO stage 2 features,
enables FP16, uses ``AdamW`` optimizer and ``WarmupLR`` scheduler:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

       "zero_optimization": {
           "stage": 2,
           "allgather_partitions": true,
           "allgather_bucket_size": 5e8,
           "overlap_comm": true,
           "reduce_scatter": true,
           "reduce_bucket_size": 5e8,
           "contiguous_gradients": true,
           "cpu_offload": true
       },

       "optimizer": {
         "type": "AdamW",
         "params": {
           "lr": 3e-5,
           "betas": [ 0.8, 0.999 ],
           "eps": 1e-8,
           "weight_decay": 3e-7
         }
       },

       "scheduler": {
         "type": "WarmupLR",
         "params": {
           "warmup_min_lr": 0,
           "warmup_max_lr": 3e-5,
           "warmup_num_steps": 500
         }
       }
    }

When you execute the program, DeepSpeed will log the configuration it received from the :class:`~transformers.Trainer`
to the console, so you can see exactly what was the final configuration passed to it.


Passing Configuration
=======================================================================================================================

As discussed in this document normally the DeepSpeed configuration is passed as a path to a json file, but if you're
not using the command line interface to configure the training, and instead instantiate the
:class:`~transformers.Trainer` via :class:`~transformers.TrainingArguments` then for the ``deepspeed`` argument you can
pass a nested ``dict``. This allows you to create the configuration on the fly and doesn't require you to write it to
the file system before passing it to :class:`~transformers.TrainingArguments`.

To summarize you can do:

.. code-block:: python

    TrainingArguments(..., deespeed="/path/to/ds_config.json")

or:

.. code-block:: python

    ds_config_dict=dict(scheduler=scheduler_params, optimizer=optimizer_params)
    TrainingArguments(..., deespeed=ds_config_dict)



Shared Configuration
=======================================================================================================================

Some configuration information is required by both the :class:`~transformers.Trainer` and DeepSpeed to function
correctly, therefore, to prevent conflicting definitions, which could lead to hard to detect errors, we chose to
configure those via the :class:`~transformers.Trainer` command line arguments.

Therefore, the following DeepSpeed configuration params shouldn't be used with the :class:`~transformers.Trainer`:

* ``train_batch_size``
* ``train_micro_batch_size_per_gpu``
* ``gradient_accumulation_steps``

as these will be automatically derived from the run time environment and the following 2 command line arguments:

.. code-block:: bash

    --per_device_train_batch_size 8 --gradient_accumulation_steps 2

which are always required to be supplied.

Of course, you will need to adjust the values in this example to your situation.



ZeRO
=======================================================================================================================

`Zero Redundancy Optimizer (ZeRO) <https://www.deepspeed.ai/tutorials/zero/>`__ is the work horse of DeepSpeed. It
support 3 different levels (stages) of optimization. The first one is not quite interesting for scalability purposes,
therefore this document focuses on stages 2 and 3. You will find more indepth information in the DeepSpeed
documentation.

The ``zero_optimization`` section of the configuration file is the most important part (`docs
<https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`__), since that is where you define
which ZeRO stages you want to enable and how to configure them. You will find the explanation for each parameter in the
DeepSpeed docs.

This section has to be configured exclusively via DeepSpeed configuration - the :class:`~transformers.Trainer` provides
no equivalent command line arguments.

Note: currently DeepSpeed doesn't validate parameter names, so if you misspell any, it'll use the default setting for
the parameter that got misspelled. You can watch the DeepSpeed engine start up log messages to see what values it is
going to use.


ZeRO-2 Config
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following is an example configuration for ZeRO stage 2:

.. code-block:: json

    {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": true,
            "allgather_bucket_size": 5e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": true,
            "cpu_offload": true
        }
    }

**Performance tuning:**

- enabling ``cpu_offload`` should reduce GPU RAM usage (it requires ``"stage": 2``)
- ``"overlap_comm": true`` trades off increased GPU RAM usage to lower all-reduce latency. ``overlap_comm`` uses 4.5x
  the ``allgather_bucket_size`` and ``reduce_bucket_size`` values. So if they are set to 5e8, this requires a 9GB
  footprint (``5e8 x 2Bytes x 2 x 4.5``). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting
  OOM-errors you will need to reduce those parameters to about ``2e8``, which would require 3.6GB. You will want to do
  the same on larger capacity GPU as well, if you're starting to hit OOM.
- when reducing these buffers you're trading communication speed to avail more GPU RAM. The smaller the buffer size,
  the slower the communication, and the more GPU RAM will be available to other tasks. So if a bigger batch size is
  important, getting a slightly slower training time could be a good trade.


ZeRO-3 Config
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following is an example configuration for ZeRO stage 3:


.. code-block:: json

    {
        "zero_optimization": {
            "stage": 3,
            "cpu_offload": true,
            "cpu_offload_params": true,
            "cpu_offload_use_pin_memory" : true,
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e14,
            "reduce_bucket_size": 1e6,
            "stage3_prefetch_bucket_size": 0.94e6,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        }
    }

Note: if you're migrating from ZeRO-2 configuration that: ``allgather_partitions``, ``allgather_bucket_size`` and
``reduce_scatter`` configuration parameters are not used in ZeRO-3. If you keep these they will just be ignored.

**Performance tuning:**

- ``sub_group_size``: ``1e14``
- ``reduce_bucket_size``: ``hidden_size*hidden_size``
- ``stage3_prefetch_bucket_size``: ``0.9 * hidden_size * hidden_size``
- ``stage3_param_persistence_threshold``: ``10 * hidden_size``
- ``stage3_max_live_parameters``: ``1e9``
- ``stage3_max_reuse_distance``: ``1e9``

If hitting OOM reduce ``stage3_max_live_parameters`` and ``stage3_max_reuse_distance``. They should have minimal impact
on performance unless you are doing activation checkpointing. ``1e9`` would consume ~2GB. The memory is shared by
``stage3_max_live_parameters`` and ``stage3_max_reuse_distance``, so its not additive, its just 2GB total.

``stage3_max_live_parameters`` is the upper limit on how many full parameters you want to keep on the GPU at any given
time. "reuse distance" is a metric we are using to figure out when will a parameter be used again in the future, and we
use the ``stage3_max_reuse_distance`` to decide whether to throw away the parameter or to keep it. If a parameter is
going to be used again in near future (less than ``stage3_max_reuse_distance``) then we keep it to reduce communication
overhead. This is super helpful when you have activation checkpointing enabled, where we do a forward recompute and
backward passes a a single layer granularity and want to keep the parameter in the forward recompute till the backward

If you set ``reduce_bucket_size``, ``stage3_prefetch_bucket_size`` and ``stage3_param_persistence_threshold`` as
recommended above, they will already be fairly small so you won't have to tune those much.

Since ``hidden_size`` varies from model to model, the ``Trainer`` will automatically set the needed value for the 3
config parameters that contain that variable (using ``model.config.hidden_size``). Just set these values to ``0`` as
shown below and the right configuration will be passed to DeepSpeed:

.. code-block:: json

    {
        "zero_optimization": {
            "stage": 3,
            "cpu_offload": true,
            "cpu_offload_params": true,
            "cpu_offload_use_pin_memory" : true,
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e14,
            "reduce_bucket_size": 0,
            "stage3_prefetch_bucket_size": 0,
            "stage3_param_persistence_threshold": 0,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        }
    }

``stage3_gather_fp16_weights_on_model_save`` enables model fp16 weights consolidation when model gets saved. With large
models and multiple GPUs this is an expensive operation both in terms of memory and speed. It's currently required if
you plan to resume the training. Watch out for future updates that will remove this limitation and make things more
flexible.


ZeRO-2 vs ZeRO-3 Performance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ZeRO-3 is likely to be slower than ZeRO-2 if everything else is configured the same because the former has to gather
model weights in addition to what ZeRO-2 does. If ZeRO-2 meets your needs and you don't need to scale beyond a few GPUs
then you may choose to stick to it. It's important to understand that ZeRO-3 enables a much higher scalability capacity
at a cost of speed.

It's possible to adjust ZeRO-3 configuration to make it perform closer to ZeRO-2:

- set ``stage3_param_persistence_threshold`` to a very large number - larger than the largest parameter, e.g., ``6 *
  hidden_size * hidden_size``. This will keep the parameters on the GPUs.
- turn off ``cpu_offload_params`` since ZeRO-2 doesn't have that option.

The performance will likely improve significantly with just ``cpu_offload_params`` turned off, even if you don't change
``stage3_param_persistence_threshold``. Of course, these changes will impact the size of the model you can train. So
these help you to trade scalability for speed depending on your needs.



ZeRO-2 Example
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here is a full ZeRO-2 all-enabled configuration file ``ds_config_zero2.json``:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": true,
            "allgather_bucket_size": 2e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true,
            "cpu_offload": true
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500
            }
        },

        "steps_per_print": 2000,
        "wall_clock_breakdown": false
    }



ZeRO-3 Example
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here is a full ZeRO-3 all-enabled configuration file ``ds_config_zero3.json``:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "zero_optimization": {
            "stage": 3,
            "cpu_offload": true,
            "cpu_offload_params": true,
            "cpu_offload_use_pin_memory" : true,
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e14,
            "reduce_bucket_size": 1e6,
            "stage3_prefetch_bucket_size": 0.94e6,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500
            }
        },

        "steps_per_print": 2000,
        "wall_clock_breakdown": false
    }


Optimizer and Scheduler
=======================================================================================================================

As long as you don't enable ``cpu_offload`` you can mix and match DeepSpeed and HuggingFace schedulers and optimizers,
with the exception of using the combination of HuggingFace scheduler and DeepSpeed optimizer:

+--------------+--------------+--------------+
| Combos       | HF Scheduler | DS Scheduler |
+--------------+--------------+--------------+
| HF Optimizer | Yes          | Yes          |
+--------------+--------------+--------------+
| DS Optimizer | No           | Yes          |
+--------------+--------------+--------------+

If ``cpu_offload`` is enabled you must use both DeepSpeed scheduler and DeepSpeed optimizer.



Optimizer
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


DeepSpeed's main optimizers are Adam, AdamW, OneBitAdam, and Lamb. These have been thoroughly tested with ZeRO and are
thus recommended to be used. It, however, can import other optimizers from ``torch``. The full documentation is `here
<https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`__.

If you don't configure the ``optimizer`` entry in the configuration file, the :class:`~transformers.Trainer` will
automatically set it to ``AdamW`` and will use the supplied values or the defaults for the following command line
arguments: ``--learning_rate``, ``--adam_beta1``, ``--adam_beta2``, ``--adam_epsilon`` and ``--weight_decay``.

Here is an example of the pre-configured ``optimizer`` entry for ``AdamW``:

.. code-block:: json

    {
       "optimizer": {
           "type": "AdamW",
           "params": {
             "lr": 0.001,
             "betas": [0.8, 0.999],
             "eps": 1e-8,
             "weight_decay": 3e-7
           }
         }
    }

Note that the command line arguments will override the values in the configuration file. This is so that there is one
definitive source of the values and to avoid hard to find errors when for example, the learning rate is set to
different values in different places. Command line rules. The values that get overridden are:

- ``lr`` with the value of ``--learning_rate``
- ``betas`` with the value of ``--adam_beta1 --adam_beta2``
- ``eps`` with the value of ``--adam_epsilon``
- ``weight_decay`` with the value of ``--weight_decay``

Therefore please remember to tune the shared hyperparameters on the command line.

If you want to use another optimizer which is not listed above, you will have to add ``"zero_allow_untested_optimizer":
true`` to the top level configuration.

If you want to use one of the officially supported optimizers, configure them explicitly in the configuration file, and
make sure to adjust the values. e.g. if use Adam you will want ``weight_decay`` around ``0.01``.


Scheduler
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

DeepSpeed supports LRRangeTest, OneCycle, WarmupLR and WarmupDecayLR LR schedulers. The full documentation is `here
<https://www.deepspeed.ai/docs/config-json/#scheduler-parameters>`__.


Here is where the schedulers overlap between 🤗 Transformers and DeepSpeed:

* ``WarmupLR`` via ``--lr_scheduler_type constant_with_warmup``
* ``WarmupDecayLR`` via ``--lr_scheduler_type linear``. This is also the default value for ``--lr_scheduler_type``,
  therefore, if you don't configure the scheduler this is scheduler that will get configured by default.


If you don't configure the ``scheduler`` entry in the configuration file, the :class:`~transformers.Trainer` will use
the values of ``--lr_scheduler_type``, ``--learning_rate`` and ``--warmup_steps`` to configure a 🤗 Transformers version
of it.

Here is an example of the pre-configured ``scheduler`` entry for ``WarmupLR``:

.. code-block:: json

    {
       "scheduler": {
             "type": "WarmupLR",
             "params": {
                 "warmup_min_lr": 0,
                 "warmup_max_lr": 0.001,
                 "warmup_num_steps": 1000
             }
         }
    }

Note that the command line arguments will override the values in the configuration file. This is so that there is one
definitive source of the values and to avoid hard to find errors when for example, the learning rate is set to
different values in different places. Command line rules. The values that get overridden are:

- ``warmup_max_lr`` with the value of ``--learning_rate``
- ``warmup_num_steps`` with the value of ``--warmup_steps``
- ``total_num_steps`` with either the value of ``--max_steps`` or if it is not provided, derived automatically at run
  time based on the environment and the size of the dataset and other command line arguments (needed for
  ``WarmupDecayLR``).

Therefore please remember to tune the shared hyperparameters on the command line.

For example, for ``WarmupDecayLR``, you can use the following entry:

.. code-block:: json

    {
       "scheduler": {
             "type": "WarmupDecayLR",
             "params": {
                 "total_num_steps": 10,
                 "last_batch_iteration": -1,
                 "warmup_min_lr": 0,
                 "warmup_max_lr": 0.001,
                 "warmup_num_steps": 1000
             }
         }
    }

and ``warmup_max_lr``, ``warmup_num_steps`` and ``total_num_steps`` will be corrected at loading time.



Automatic Mixed Precision
=======================================================================================================================

You can use automatic mixed precision with either a pytorch-like AMP way or the apex-like way:

If you want to use an equivalent of the Pytorch native amp, you can either configure the ``fp16`` entry in the
configuration file, or use the following command line arguments: ``--fp16 --fp16_backend amp``.

Here is an example of the ``fp16`` configuration:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
    }

Here is the `documentation <https://www.deepspeed.ai/docs/config-json/#fp16-training-options>`__.

If you want to use NVIDIA's apex instead, you can can either configure the ``amp`` entry in the configuration file, or
use the following command line arguments: ``--fp16 --fp16_backend apex --fp16_opt_level 01``.

Here is an example of the ``amp`` configuration:

.. code-block:: json

    {
        "amp": {
            "enabled": true,
            "opt_level": "O1"
        }
    }

Here is the `documentation
<https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options>`__.


Gradient Accumulation
=======================================================================================================================

While normally DeepSpeed gets gradient accumulation configured with:

.. code-block:: json

    {
        "gradient_accumulation_steps": 3,
    }

in this case, to enable gradient accumulation, pass the command line ``--gradient_accumulation_steps 3`` argument as
normal and it will get injected into the DeepSpeed configuration.

If you try to add it directly to the configuration file, you will receive an error from the ``Trainer`` - this is
because this setting is needed by the ``Trainer`` too, and so this approach ensures that there is a single way of
setting this value and thus avoid potential subtle errors.






Gradient Clipping
=======================================================================================================================

If you don't configure the ``gradient_clipping`` entry in the configuration file, the :class:`~transformers.Trainer`
will use the value of the ``--max_grad_norm`` command line argument to set it.

Here is an example of the ``gradient_clipping`` configuration:

.. code-block:: json

    {
        "gradient_clipping": 1.0,
    }



Getting the model weights out
=======================================================================================================================

As long as you continue training and resuming using DeepSpeed you don't need to worry about anything. DeepSpeed stores
fp32 master weights in its custom checkpoint optimizer files, which are ``global_step*/*optim_states.pt`` (this is glob
pattern), and are saved under the normal checkpoint.

**FP16 Weights:**

When a model is saved under ZeRO-2, you end up having the normal ``pytorch_model.bin`` file with the model weights, but
they are only the fp16 version of the weights.

Under ZeRO-3, things are much more complicated, since the model weights are partitioned out over multiple GPUs,
therefore ``"stage3_gather_fp16_weights_on_model_save": true`` is required to get the ``Trainer`` to save the fp16
version of the weights. If this setting is ``False`` ``pytorch_model.bin`` won't be created. This is because by default
DeepSpeed's ``state_dict`` contains a placeholder and not the real weights. If we were to save this ``state_dict`` it
won't be possible to load it back.

**FP32 Weights:**

While the fp16 weights are fine for resuming training, if you finished finetuning your model and want to upload it to
the `models hub <https://huggingface.co/models>`__ or pass it to someone else you most likely will want to get the fp32
weights. This cannot be done during training since this is a process that requires a lot of memory, and therefore this
is performed offline.

DeepSpeed creates a special conversion script ``zero_to_fp32.py`` which it places in the top-level of the checkpoint
folder. Using this script you can extract the weights at any point. The script is standalone and you no longer need to
have the configuration file or a ``Trainer`` to do the extraction.

Let's say your checkpoint folder looks like this:

.. code-block:: bash

    $ ls -l output_dir/checkpoint-1/
    -rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
    drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
    -rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
    -rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
    -rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
    -rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
    -rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
    -rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
    -rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
    -rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
    -rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
    -rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*

In this example there is just one DeepSpeed checkpoint sub-folder `global_step1`. Therefore to reconstruct the fp32
weights just run:

.. code-block:: bash

    python zero_to_fp32.py global_step1 pytorch_model.bin

The script will automatically handle either ZeRO-2 or ZeRO-3 checkpoint.

``python zero_to_fp32.py -h`` will give you usage details.

If you have multiple DeepSpeed checkpoint sub-folders, pick the one you know to have the desired weights.

This is it. ``pytorch_model.bin`` will now contain the full fp32 model weights consolidated from multiple GPUs.

Note: currently the script requires 2x general RAM of the final fp32 model weights.

ZeRO 3 Nuances
=======================================================================================================================

ZeRO 3 is quite different from ZeRO 2 because of its param sharding feature.

While all the efforts were made for things to just work without needing any special changes to your models, in certain
circumstances you may find the following information to be needed.


Registering External Parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If layer A needs to access weights belonging to layer B, currently layer A needs to tell DeepSpeed about it. This is
done with the help of ``deepspeed.zero.register_external_parameter`` that needs to be called in ``A.__init__`` and can
be seen in the following example:

.. code-block:: python

    class ModuleZ3(torch.nn.Module):
        def __init__(self, *args):
            super().__init__(self, *args)
            self.layer1 = SomeLayer()
            self.layer2 = OtherLayer()
            deepspeed.zero.register_external_parameter(self, self.layer1.weight)

        def forward(self, input):
            x = self.layer1(input)
            # self.layer1.weight is needed in ModuleZ3.forward
            y = self.layer2(x, self.layer1.weight)
            return y

In general ``transformers`` models don't use this style of referring to other layer's weights so most likely you won't
need to use it.

For full details on this method please refer to `Registering External Parameters
<https://deepspeed.readthedocs.io/en/latest/zero3.html#registering-external-parameters>`__.



Constructing Massive Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

DeepSpeed/ZeRO-3 can handle models with Trillions of parameters which may not fit onto the existing RAM. In such cases,
but also if you want the initialization to happen much faster, initialize the model using `deepspeed.zero.Init()`
context manager (which is also a function decorator), like so:

.. code-block:: python

    from transformers import T5ForConditionalGeneration, T5Config
    import deepspeed
    with deepspeed.zero.Init():
       config = T5Config.from_pretrained("t5-small")
       model = T5ForConditionalGeneration(config)

As you can see this gives you a randomly initialized model.

If you want to use a pretrained model, ``model_class.from_pretrained`` will activate this feature as long as
``is_deepspeed_zero3_enabled()`` returns ``True``, which can be set manually via ``deepspeed_zero3_enable(True)``.
Therefore to enable this feature here is the required sequence:

.. code-block:: python

    from transformers.integrations import deepspeed_zero3_enable
    deepspeed_zero3_enable(True)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

If you're using ``Trainer`` command line arguments which include ``--deepspeed ds_config.json`` with ZeRO-3 config
enabled, then you can skip ``deepspeed_zero3_enable(True)`` as it will try to discover whether it'll be run under
ZeRO-3 and ``from_pretrained`` will automatically activate this feature.

Note: If the fp16 weights of the model can't fit onto the memory of a single GPU this feature must be used.

For full details on this method and other related features please refer to `Constructing Massive Models
<https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models>`__.





Gathering Parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Under ZeRO-3 on multiple GPUs no single GPU has all the parameters unless it's the parameters for the currently
executing layer. So if you need to access all parameters from all layers at once there is a specific method to do it.
Most likely you won't need it, but if you do please refer to `Gathering Parameters
<https://deepspeed.readthedocs.io/en/latest/zero3.html#manual-parameter-coordination>`__

We do however use it internally in several places, one such example is when loading pretrained model weights in
``from_pretrained``. We load one layer at a time and immediately partition it to all participating GPUs, as for very
large models it won't be possible to load it on one GPU and then spread it out to multiple GPUs, due to memory
limitations.

Also under ZeRO-3, if you write your own code and run into a model parameter weight that looks like:

.. code-block:: python

    tensor([1.], device='cuda:0', dtype=torch.float16, requires_grad=True)

stress on ``tensor([1.])``, or if you get an error where it says the parameter is of size ``1``, instead of some much
larger multi-dimensional shape, this means that the parameter is partitioned and what you see is a ZeRO-3 placeholder.





Notes
=======================================================================================================================

* DeepSpeed works with the PyTorch :class:`~transformers.Trainer` but not TF :class:`~transformers.TFTrainer`.
* While DeepSpeed has a pip installable PyPI package, it is highly recommended that it gets installed from `source
  <https://github.com/microsoft/deepspeed#installation>`__ to best match your hardware and also if you need to enable
  certain features, like 1-bit Adam, which aren't available in the pypi distribution.
* You don't have to use the :class:`~transformers.Trainer` to use DeepSpeed with 🤗 Transformers - you can use any model
  with your own trainer, and you will have to adapt the latter according to `the DeepSpeed integration instructions
  <https://www.deepspeed.ai/getting-started/#writing-deepspeed-models>`__.

Main DeepSpeed Resources
=======================================================================================================================

- `Project's github <https://github.com/microsoft/deepspeed>`__
- `Usage docs <https://www.deepspeed.ai/getting-started/>`__
- `API docs <https://deepspeed.readthedocs.io/en/latest/index.html>`__
- `Blog posts <https://www.microsoft.com/en-us/research/search/?q=deepspeed>`__

Papers:

- `ZeRO: Memory Optimizations Toward Training Trillion Parameter Models <https://arxiv.org/abs/1910.02054>`__
- `ZeRO-Offload: Democratizing Billion-Scale Model Training <https://arxiv.org/abs/2101.06840>`__

Finally, please, remember that, HuggingFace :class:`~transformers.Trainer` only integrates DeepSpeed, therefore if you
have any problems or questions with regards to DeepSpeed usage, please, file an issue with `DeepSpeed GitHub
<https://github.com/microsoft/DeepSpeed/issues>`__.
