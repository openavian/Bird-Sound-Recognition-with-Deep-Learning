# Large-Scale-Birds-Recognition-in-China

This repo is my Signature Work @ Duke Kunshan University. We aim to build a pipeline for Large Scale Birds Detection and Recognition in China.

- For bird song **data retrieval**, refer to our toolkit [**XenoPy**](https://github.com/realzza/xenopy) build upon Xeno-canto API 2.0.

- For bird song **detection**, take a look at our latest release [**easybird**](https://github.com/realzza/easybird) library.

- For bird song **recognition**, refer to the following usage for now. The [**recognition** module](recognition.py) will be integrated to the easybird library in the near future.

## Setup
```bash
pip install -r requirements.txt
```

## Demo
```python
from recognition import from_wav, from_wavs
```

#### `from_wav`
```python
from_wav('sample_wavs/AmurFalcon.wav')
```
```python
>>> ('AmurFalcon', 0.9992602467536926)
```

#### `from_wavs`
```python
from_wavs(['sample_wavs/AmurFalcon.wav','sample_wavs/BarnacleGoose.wav'])
```
```python
>>> [('AmurFalcon', 'AmurFalcon', 0.9992602467536926),
 ('BarnacleGoose', 'BarnacleGoose', 0.9826954007148743)]
```

## Open Source
We welcome the community to contribute to our work! File an [issue](https://github.com/realzza/Large-Scale-Birds-Recognition-in-China/issues) had you encountered any bugs, or submit a PR to help us improve.