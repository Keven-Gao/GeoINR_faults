Installation
============

GeoINR supports Python 3.8+ and can be installed with either ``pip`` or ``conda``.

Pip Installation
------------------------------

.. raw:: html

   <div class="install-command-card">
     <code id="install-cmd-pip">pip install geoinr_faults</code>
     <button class="install-copy-btn" data-copy-target="install-cmd-pip">Copy</button>
   </div>

Update ``torch`` to support GPU acceleration (optional):
------------------

The default installation of ``torch`` not include GPU support. If you have a compatible NVIDIA GPU and want to leverage GPU acceleration, you can install the appropriate version of PyTorch with CUDA support. 
Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) to find the correct command for your system. 

.. code-block:: commandline

   # Uninstall current torch version if necessary
   pip uninstall torch torchvision torchaudio
   # Install torch with CUDA support (example for CUDA 13.0)
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130


Verify Installation
-------------------

.. code-block:: python

   import geoinr_faults
   print(geoinr_faults.__version__)
