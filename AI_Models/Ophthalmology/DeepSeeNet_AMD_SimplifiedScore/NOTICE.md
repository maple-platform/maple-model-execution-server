# Attribution notice

This package contains an inference-only adaptation derived from:

- DeepSeeNet, ncbi-nlp/DeepSeeNet: https://github.com/ncbi-nlp/DeepSeeNet
- Paper: Peng Y, Dharssi S, Chen Q, Keenan T, Agron E, Wong W, Chew E, Lu Z.
  *DeepSeeNet: A deep learning model for automated classification of
  patient-based age-related macular degeneration severity from color fundus
  photographs.* Ophthalmology. 2019;126(4):565-575.

## License

Quoted in full from the upstream `LICENSE.txt`:

> PUBLIC DOMAIN NOTICE -- National Center for Biotechnology Information
>
> This software/database is a "United States Government Work" under the
> terms of the United States Copyright Act. It was written as part of the
> author's official duties as a United States Government employee and thus
> cannot be copyrighted. This software/database is freely available to the
> public for use. The National Library of Medicine and the U.S. Government
> have not placed any restriction on its use or reproduction.
>
> Although all reasonable efforts have been taken to ensure the accuracy and
> reliability of the software and data, the NLM and the U.S. Government do
> not and cannot warrant the performance or results that may be obtained by
> using this software or data. The NLM and the U.S. Government disclaim all
> warranties, express or implied, including warranties of performance,
> merchantability or fitness for any particular purpose.
>
> Please cite the author in any work or product based on this material.

Separately, the upstream README states (a usage restriction, not a copyright
term): *"The performance characteristics of this product have not been
evaluated by the Food and Drug Administration and is not intended for
commercial use or purposes beyond research use only."*

Local changes: extracted the crop-to-square / resize / `preprocess_input`
pipeline and the three risk-factor model calls (drusen, pigment, advanced
AMD) plus `get_simplified_score`'s scoring logic from
`deepseenet_simplified.py` into a single Maple `main()` entrypoint. No
changes to model architecture or weights. Geographic atrophy / central GA
models (tag `0.2`) were not adapted (out of scope; see `README.md`). No
endorsement by the original authors is implied.

## Clinical limitation

This is not a medical device and must not be used for autonomous diagnosis or
patient-care decisions. NCBI's own disclaimer: *"The information produced on
this website is not intended for direct diagnostic use or medical
decision-making without review and oversight by a clinical professional."*
