MoViAD
========

.. raw:: html

   <div class="moviad-hero">
     <span class="moviad-eyebrow">VISUAL ANOMALY DETECTION / RESEARCH TOOLKIT</span>
     <h2>Find the unexpected.<br>Keep learning.</h2>
     <p>Build visual inspection pipelines from modular models, datasets,
     feature extractors, and continual learning strategies.</p>
     <div class="moviad-actions">
       <a href="getting-started.html">Start building →</a>
       <a class="secondary" href="continual.html">Explore continual learning</a>
     </div>
   </div>
   <div class="moviad-grid">
     <a class="moviad-card" href="workflows.html"><span>01 / STANDARD</span>
       <h3>One category. A clear baseline.</h3>
       <p>Connect a feature extractor, anomaly model, dataset, and trainer.</p></a>
     <a class="moviad-card" href="continual.html"><span>02 / CONTINUAL</span>
       <h3>A stream of tasks.</h3>
       <p>Fine tune, replay, or maintain model-specific statistics over time.</p></a>
     <a class="moviad-card" href="reference.html"><span>03 / MODULAR</span>
       <h3>Choose your components.</h3>
       <p>Understand the contracts behind training and evaluation.</p></a>
   </div>

Built for experimentation
---------------------------

MoViAD is a research library for visual anomaly detection, including industrial
inspection and edge-oriented feature extraction. The current implementation
lives under ``src/moviad``. Standard and continual workflows share dataset,
training, and evaluation components. Model-specific and legacy paths have
different levels of maturity; see :doc:`migration` before using older scripts.

.. toctree::
   :maxdepth: 2
   :caption: Learn & build

   getting-started
   workflows
   continual
   reference
   migration
   contributing
