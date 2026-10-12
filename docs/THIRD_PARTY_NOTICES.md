# Third-party notices

acestep.cpp is betweentwomidnights' fork descended from
[twilwa/Sawhee](https://github.com/twilwa/sawhee) and
[ServeurpersoCom/acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp).
The project is distributed under the included MIT `LICENSE`, preserving that
history and the original authors' notices in the source.

The runtime contains the following third-party code. Model weights are not
included; their licenses and download terms are separate.

| Component | License | Notice in the archive/source |
| --- | --- | --- |
| betweentwomidnights/ggml and upstream ggml contributors | MIT | `LICENSE-ggml.txt` |
| cpp-httplib, Yuji Hirose and contributors | MIT | `LICENSE-cpp-httplib.txt` |
| yyjson, YaoYuan and contributors | MIT | `LICENSE-yyjson.txt` |
| minimp3, lieff and contributors | CC0 1.0 | CC0 dedication retained in `minimp3-source.h` |
| mp3 encoder/decoder and WAV/audio helpers in this tree | Notices retained in source | Project `LICENSE` and original file headers |

The Windows standalone archive additionally redistributes NVIDIA CUDA runtime
and cuBLAS libraries under `NVIDIA-CUDA-EULA.txt`. The split CUDA backend archive
contains only `ggml-cuda.dll`; the application supplies its separately installed
compatible CUDA runtime.
