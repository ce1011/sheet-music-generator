import flet as ft
import subprocess
import os

import copy
from types import SimpleNamespace

import numpy as np, scipy, matplotlib.pyplot as plt, matplotlib
import librosa, librosa.display
import madmom
import music21
import drumsnotes

# YOLO V8
from ultralytics import YOLO
from PIL import Image
import cv2

cls_to_object = {
    0: "Kick",
    1: "Snare",
    2: "HiHat",
}

pitch_to_position_map = {
    36: 'Kick',
    38: 'SnareHead',
    40: 'SnareRim',
    37: 'SnareXStick',
    48: 'HighTom',
    50: 'HighTomRim',
    45: 'LowMidTom',
    47: 'LowMidTomRim',
    43: 'HighFloorTom',
    58: 'HighFloorTomRim',
    46: 'OpenHiHatBow',
    26: 'OpenHiHatEdge',
    42: 'ClosedHiHatBow',
    22: 'ClosedHiHatEdge',
    44: 'ClosedHiHatPedal',
    49: 'Crash1Bow',
    55: 'Crash1Edge',
    57: 'Crash2Bow',
    52: 'Crash2Edge',
    51: 'RideBow',
    59: 'RideEdge',
    53: 'RideBell',
}

# yolo_result_to_note_map = {
#     'Kick': music21.note.Note(36),
#     'SnareHead': music21.note.Note(38),
#     'SnareRim': music21.note.Note(40),
#     'SnareXStick': music21.note.Note(37),
#     'HighTom': music21.note.Note(48),
#     'HighTomRim': music21.note.Note(50),
#     'LowMidTom': music21.note.Note(45),
#     'LowMidTomRim': music21.note.Note(47),
#     'HighFloorTom': music21.note.Note(43),
#     'HighFloorTomRim': music21.note.Note(58),
#     'OpenHiHatBow': music21.note.Note(46),
#     'OpenHiHatEdge': music21.note.Note(26),
#     'ClosedHiHatBow': music21.note.Note(42),
#     'ClosedHiHatEdge': music21.note.Note(22),
#     'ClosedHiHatPedal': music21.note.Note(44),
#     'Crash1Bow': music21.note.Note(49),
#     'Crash1Edge': music21.note.Note(55),
#     'Crash2Bow': music21.note.Note(57),
#     'Crash2Edge': music21.note.Note(52),
#     'RideBow': music21.note.Note(51),
#     'RideEdge': music21.note.Note(59),
#     'RideBell': music21.note.Note(53),
# }

# yolo_result_to_note_map = {
#     'Kick': drumsnotes.Kick(),
#     'SnareHead': drumsnotes.Snare(),
#     'SnareRim': drumsnotes.Snare(),
#     'SnareXStick': drumsnotes.Snare(),
#     'HighTom': drumsnotes.HighTom(),
#     'HighTomRim': drumsnotes.HighTom(),
#     'LowMidTom': drumsnotes.MiddleTom(),
#     'LowMidTomRim': drumsnotes.MiddleTom(),
#     'HighFloorTom': drumsnotes.LowTom(),
#     'HighFloorTomRim': drumsnotes.LowTom(),
#     'OpenHiHatBow': drumsnotes.OpenHiHat(),
#     'OpenHiHatEdge': drumsnotes.OpenHiHat(),
#     'ClosedHiHatBow': drumsnotes.HiHat(),
#     'ClosedHiHatEdge': drumsnotes.HiHat(),
#     'ClosedHiHatPedal': drumsnotes.PedalHiHat(),
#     'Crash1Bow': drumsnotes.Crash(),
#     'Crash1Edge': drumsnotes.Crash(),
#     'Crash2Bow': drumsnotes.Crash(),
#     'Crash2Edge': drumsnotes.Crash(),
#     'RideBow': drumsnotes.Ride(),
#     'RideEdge': drumsnotes.Ride(),
#     'RideBell': drumsnotes.Ride(),
# }

yolo_result_to_note_map = {
    'Kick': drumsnotes.Kick(),
    'Snare': drumsnotes.Snare(),
    'HiHat': drumsnotes.HiHat(),
}

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['figure.dpi'] = 128

cache_onset_slice_spec_folder = os.getcwd()

model = YOLO(
    r"G:\sheet-music-generating\music-transcription-gui\best.pt")


def main(page: ft.Page):
    def check_item_clicked(e):
        e.control.checked = not e.control.checked
        page.update()

    space = ft.Container(
        padding=30,
    )

    def show_snackbar(message, color):
        page.snack_bar = ft.SnackBar(ft.Text(f"{message}"), bgcolor=color)
        page.snack_bar.open = True
        page.update()

    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        selectedFilePath.value = (e.files[0].path)
        print(selectedFilePath)
        selected_files.update()

    def submit(e):
        if selected_files.value == "Cancelled!" or selected_files.value == "":
            show_snackbar("Please select a file first", ft.colors.RED_500)
        else:
            file_path_windows = selectedFilePath.value
            file_path_windows = file_path_windows.replace('\\','/')
            page.go('/processing')
            processing_page_view.append(ft.Column(
                [ft.Text("Step 1. Separating Music Source"), ft.ProgressRing(), console_output],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ), )
            # Call demucs
            process = subprocess.Popen(
                ["demucs", file_path_windows, "--mp3", "-n", "htdemucs", "-d", "cuda"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            while True:  # Could be more pythonic with := in Python3.8+
                line = process.stdout.readline()
                err_line = process.stderr.readline()
                if not line and process.poll() is not None:
                    console_output.value = console_output.value + ('Done separating')
                    page.update()
                    break
                print(line.decode())
                console_output.value = console_output.value + line.decode()
                page.update()

            # Call YOLO
            file_path_windows = os.getcwd() + '/separated/htdemucs/' + file_path_windows.split('/')[-1].split('.')[0] + '/drums.mp3'

            current_process = ft.Text("")
            process_bar_music_transcript = ft.ProgressBar(width=600, color="amber", bgcolor="#eeeeee")

            processing_page_view.append(ft.Text("Step 2. Music Transcripting")
                                    )
            processing_page_view.append(process_bar_music_transcript)
            processing_page_view.append(current_process)
            page.update()
            y, sr = librosa.load(file_path_windows, duration=63)
            plt.axis('off')
            plt.tight_layout()
            hop_length = 512
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
            # print(onset_frames)   frame numbers of estimated onsets
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            duration_of_each_slice = 60 / tempo
            index = 1

            previous_beat_no = 0

            # proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100)
            # act = madmom.features.onsets.CNNOnsetProcessor()(file_path_windows)
            # onsets = proc(act)
            # print(onsets)

            score = music21.stream.Score()
            part = music21.stream.Part()
            # part.insert(0, music21.instrument.Woodblock())
            part.partName = 'Drums'

            stream = music21.stream.Stream()
            score.append(music21.tempo.MetronomeMark(number=tempo))
            aInstrument = music21.instrument.Instrument()
            part.insert(aInstrument)

            currentNotes = 0
            all_notes = len(onset_times)

            for i in onset_times:
                if index % 8 == 0:
                    part.append(stream)
                    stream = music21.stream.Stream()
                y, sr = librosa.load(file_path_windows, offset=i, duration=duration_of_each_slice)
                waveformSavePath = cache_onset_slice_spec_folder
                plt.figure().set_size_inches(5, 5)
                plt.axis('off')
                S = np.abs(librosa.hybrid_cqt(y, sr=sr))
                librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr)
                # for line in plt.gca().lines:
                #     line.set_linewidth(0.25)
                plt.savefig(waveformSavePath + str(index) + '.png', dpi=128)
                plt.close()
                # YOLO V7
                # result = yolo.detect(waveformSavePath + str(index) + '.png')

                # YOLO V8
                im2 = cv2.imread(waveformSavePath + str(index) + '.png')
                results = model.predict(source=im2, save=False, device='0', conf=0.2)  # save plotted images\
                result = map(lambda x: cls_to_object[int(x.item())], results[0].boxes.cls)
                result = list(set(result))
                # Split result by space

                print(result)
                if len(result) > 1:
                    drums_in_notes = []
                    for j in result:
                        drums_in_notes.append(copy.deepcopy(yolo_result_to_note_map[j]))
                    stream.append(drumsnotes.Chord(drums_in_notes))
                elif len(result) == 1:
                    stream.append(copy.deepcopy(yolo_result_to_note_map[result[0]]))

                index += 1
                currentNotes += 1
                current_process.value = (f"{currentNotes} / {all_notes}".format(currentNotes = currentNotes,all_notes = all_notes))
                process_bar_music_transcript.value = int(currentNotes) /  int(all_notes)
                page.update()

            processing_page_view.append(ft.Column(
                controls=[ft.Text("Step 3. Write to MusicXML")],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ), )

            page.update()

            score.append(part)
            score.write('xml', fp=r'drums.xml')

            processing_page_view.append(ft.Column(
                controls=[ft.Text("Music transcription Done")],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ), )
            page.update()

    console_output = ft.Text()
    console_output.value = ""

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text(color=ft.colors.BROWN_200)
    selectedFilePath = ft.Text(size=0)
    selectedFilePath.value = "Cancelled!"

    page.overlay.append(pick_files_dialog)

    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = 'light'

    processing_page_view = list()

    processing_page_view.append(
        ft.ElevatedButton("Go Home", on_click=lambda _: page.go("/")),
    )

    def route_change(route):
        page.views.clear()
        page.views.append(
            ft.View(
                "/",
                [ft.AppBar(
                    leading=ft.Icon(ft.icons.MUSIC_NOTE, color=ft.colors.WHITE),
                    leading_width=40,
                    title=ft.Text("YOLO Music Transcription GUI", color=ft.colors.WHITE),
                    center_title=False,
                    bgcolor=ft.colors.BLUE_500,
                ), ft.Row(controls=[
                    ft.Column(controls=[ft.Icon(name=ft.icons.MY_LIBRARY_MUSIC, color=ft.colors.BROWN, size=250),
                                        ft.Text("Music transcription", color=ft.colors.BROWN, size=30),
                                        space,
                                        ft.Row(controls=[ft.ElevatedButton(
                                            "Pick files",
                                            width=150,
                                            icon=ft.icons.FILE_OPEN,
                                            color=ft.colors.WHITE,
                                            bgcolor=ft.colors.BROWN,
                                            on_click=lambda _: pick_files_dialog.pick_files(
                                                allow_multiple=False
                                            ),
                                        ),
                                            ft.ElevatedButton(
                                                "Submit",
                                                width=150,
                                                icon=ft.icons.UPLOAD_FILE,
                                                color=ft.colors.WHITE,
                                                bgcolor=ft.colors.BROWN,
                                                on_click=submit
                                            ), ], alignment=ft.MainAxisAlignment.CENTER),

                                        selected_files, ], alignment=ft.MainAxisAlignment.CENTER), ],
                    alignment=ft.MainAxisAlignment.CENTER)]

            )
        )
        if page.route == "/processing":
            page.views.append(
                ft.View(
                    "/processing",
                    controls=processing_page_view
                )
            )
        page.update()

    page.on_route_change = route_change
    page.go(page.route)


ft.app(target=main)
