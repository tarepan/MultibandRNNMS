```
# root
python preprocess.py in_dir=path/to/LJSpeech-1.1 out_dir=datasets/LJSpeech-1.1

    ## preprocess.py
    if __name__ == "__main__":
        preprocess_dataset()

        @hydra.main(config_path="univoc/config", config_name="preprocess")
        def preprocess_dataset(cfg):

            executor.submit(process_wav, wav_path, out_path, cfg.preprocess)

                def process_wav(wav_path, out_path, cfg):

                    mu_compress(..., frame_length=cfg.win_length, ...)

                        def mu_compress(wav, hop_length=200, frame_length=800, bits=8):
```
