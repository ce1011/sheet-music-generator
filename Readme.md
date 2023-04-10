# YOLO Music Transcription GUI

This is a music transcription program that uses YOLO for drum kit detection and transcribes the music into MusicXML format. The program is tested on Python 3.10.

## Installation

1. Clone this repository:
```commandline
git clone https://github.com/ce1011/sheet-music-generator.git
cd yolo-music-transcription-gui
```

2. Install required Python packages (excluding Librosa):
```commandline
pip install flet librosa matplotlib numpy scipy madmom music21 ultralytics
```


3. Clone and install the latest version of Librosa:

```commandline
git clone https://github.com/librosa/librosa.git
cd librosa
pip install -e .
cd ..
```
4. Install the `music21` plugin for drum notation (jbigdata-git, 2021): 
```commandline
git clone https://github.com/jbigdata-git/music21drums.git
cd music21drums
pip install -e .
cd ..
```

6. Place the YOLO model (e.g., `best.pt`) in the same directory as `main.py`.

## Usage

1. Run the program:
```commandline
python main.py
```


2. The GUI will open in your default web browser. Follow the on-screen instructions to select a music file and start the transcription process.

3. The transcription result will be saved as `drums.xml` in the project directory.

## Notes

- This program has been tested on Python 3.10 only. Compatibility with other versions is not guaranteed.
- The YOLO model should be provided in the same directory as `main.py`.






