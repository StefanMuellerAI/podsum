import whisper
import subprocess
import os
import time
import shutil
import glob
import streamlit as st
import json
import random
import logging
import mutagen
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

#logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def check_mp3_integrity(file_path):
    try:
        # Versucht, die Datei zu öffnen und zu analysieren
        audio_file = mutagen.File(file_path)
        # Wenn audio_file None ist, unterstützt mutagen den Dateityp nicht oder die Datei konnte nicht gelesen werden
        if audio_file is None:
            return False
        return True
    except mutagen.MutagenError:
        # Fängt spezifische Fehler ab, die während der Analyse der Datei auftreten können
        return False
def transcribe_podcast(file_path, output_file):
    model = whisper.load_model("base")  # Oder 'small', 'medium', 'large', je nach Bedarf
    result = model.transcribe(file_path)

    with open(output_file, "w") as f:
        for i, segment in enumerate(result["segments"], start=1):
            f.write(f"Segment {i}: {segment['start']}-{segment['end']}: {segment['text']}\n")

# Verwenden Sie diese Funktion, um Ihren Podcast zu transkribieren und in eine Datei zu schreiben

def apply_fade_to_segments(segments_folder, session_id):
    # Funktion zum Überprüfen, ob die spezifische Nummer im Dateinamen enthalten ist
    def contains_specific_number(filename, number):
        return str(number) in filename

    # Liste der Segment-Dateien im angegebenen Ordner
    segment_files = glob.glob(os.path.join(segments_folder, '*.mp3'))

    faded_files = []
    for segment_file in segment_files:
        if contains_specific_number(segment_file, session_id):
            faded_file = segment_file.replace(".mp3", "_faded.mp3")
            subprocess.run([
                "ffmpeg",
                "-i", segment_file,
                "-af", "afade=t=in:st=0:d=0.5",
                "-acodec", "libmp3lame",
                "-loglevel", "debug",
                faded_file
            ])
            faded_files.append(faded_file)

    return faded_files

def merge_mp3_with_separator(segments_folder, file_path_separator, output_file, session_id, file_path_intro):
    # Löschen der spezifischen Datei, falls vorhanden
    specific_file_to_delete = os.path.join(segments_folder, f"{session_id}_output_segment_0.mp3")
    if os.path.exists(specific_file_to_delete):
        os.remove(specific_file_to_delete)

    # Anwenden von Fade-In und Fade-Out auf jedes Segment
    segment_files = apply_fade_to_segments(segments_folder, session_id)

    # Erstellen einer temporären Datei, die das Intro, alle Segmente und Trenner kombiniert
    with open(f"{session_id}_temp_filelist.txt", "w") as filelist:
        # Zuerst die Intro-Datei hinzufügen
        if os.path.exists(file_path_intro):
            filelist.write(f"file '{file_path_intro}'\n")

        for i, segment_file in enumerate(segment_files):
            if i > 0:  # Vor jedes Segment nach dem ersten einen Trenner einfügen
                filelist.write(f"file '{file_path_separator}'\n")
            filelist.write(f"file '{segment_file}'\n")

    # FFmpeg-Befehl zum Zusammenführen der Dateien
    subprocess.run([
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", f"{session_id}_temp_filelist.txt",
        "-c", "copy",
        "-acodec", "libmp3lame",
        "-loglevel", "debug",
        output_file
    ])

    # Löschen der temporären Dateiliste und der gefadeten Dateien
    os.remove(f"{session_id}_temp_filelist.txt")
    for file in segment_files:
        os.remove(file)

def get_type_and_topic(transcript):
    print("Verarbeite nächsten Block")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher KI-Assistent, der spezialisiert ist anhand eines Auszugs eines Podcast Transscripts zu erkennen, welches Thema die Episode hat und um welche Art Episode es sich handelt. Das Ergebnis deiner Analyse gibst du als JSON-Objekt zurück."},
            {"role": "user",
             "content": f"Gebe bitte als Key 'type' und als Value entweder 'solo' oder 'interview' zurück. Jenachdem ob es nur einen Sprecher oder mehrere Sprecher gibt. Außerdem gib bitte unter Key 'topic' und value das Thema der Podcast-Episode in maximal 5 Worten zurück. Hier ist der Auszug aus dem Transkript. Ignoriere bitte Segmente und Zeitstempel: {transcript}"}
        ],
        max_tokens=1000,
        temperature=1,
        response_format={"type": "json_object"}
    )
    # Prüfen, ob die Antwort ein gültiges JSON-Objekt ist
    response_json = response.choices[0].message.content
    data = json.loads(response_json)

    # Extraktion der Werte für 'type' und 'topic'
    type = data.get("type", "Unbekannter Typ")
    topic = data.get("topic", "Unbekanntes Thema")

    return type, topic

def select_segments(block, topic, type):
    logging.info("Segmente werden selektiert")
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher KI-Assistent, der Segmente im Transkript auswählt, die besonders hohe Relevanz anhand einer Fragestellung haben."},
            {"role": "user",
             "content": f"Identifiziere bitte jeweils mindestens 3 zusammenhängende, numerisch aufeinanderfolgende Segmente, die eine für den Zuhörenden extrem relevante Aussage zum Thema:'{topic}' des Podcasts enthalten. Es dürfen keine Eigenwerbung, Verweise auf Websites oder ähnliches enthalten sein. Zusätzlich muss es sich um vom Interviewgast gegebene Antworten handeln, sofern der Typ Interview ist. Wenn der Type 'Solo' ist, beachte bitte alle Inhalte. Das ist der Typ der Podcast Episode {type}. Gib ausschließlich die Segmentnummern per Komata getrennt zurück ohne Leerzeichen. Wenn du keine Segmente mehr findest, gib 0 aus. Hier ist das Transkript {block}"}
        ],
        max_tokens=1000,
        temperature=1,
    )
    response_text = response.choices[0].message.content
    print(response_text)
    segment_numbers = response_text.split(',')
    segment_numbers = [int(num.strip()) for num in segment_numbers if num.strip().isdigit()]
    return segment_numbers


def extract_max_1000_words(file_path_transcript):
    try:
        # Öffnen der Datei und Lesen des Inhalts
        with open(file_path_transcript, 'r', encoding='utf-8') as file:
            content = file.read()
        # Teilen des Inhalts in Worte
        words = content.split()
        # Beschränken auf maximal 1000 Worte
        words = words[:1000]
        # Zusammenfügen der Worte zu einem String
        return ' '.join(words)
    except FileNotFoundError:
        return "Die angegebene Datei wurde nicht gefunden."
    except Exception as e:
        return f"Ein Fehler ist aufgetreten: {e}"

def extract_multiple_segments_to_single_file(file_path_transcript, segment_numbers, input_file, output_dir, session_id):
    filters = []
    with open(file_path_transcript, "r") as file:
        lines = file.readlines()
    for segment_number in segment_numbers:
        for line in lines:
            if line.startswith(f"Segment {segment_number}:"):
                _, time_range_with_text = line.split(":", 1)
                time_range, _ = time_range_with_text.split(":", 1)
                start_time, end_time = time_range.strip().split("-")
                filters.append(f"between(t,{start_time},{end_time})")
    filter_string = '+'.join(filters)
    segment_numbers_str = "_".join(map(str, segment_numbers))
    print("Erstelle Segment")
    output_file = os.path.join(output_dir, f"{session_id}_output_segment_{segment_numbers_str}.mp3")
    subprocess.run([
        "ffmpeg", "-i", input_file,
        "-af", f"aselect='{filter_string}',asetpts=N/SR/TB",
        "-acodec", "libmp3lame",
        "-loglevel", "debug",
        output_file
    ])

def delete_files_with_number(folder, session_id):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # Prüfen, ob die Datei die spezifische Zahl im Namen enthält
        if str(session_id) in filename:
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            print(f"Skipped: {file_path}")
def process_transcription_in_blocks(file_path_transcript, file_path_episode, session_id, topic, type, block_size=25):
    with open(file_path_transcript, "r") as file:
        segments = file.readlines()
    print(len(segments))

    for i in range(0, len(segments), block_size):
        block = segments[i:i + block_size]
        segment_numbers = select_segments(block, topic, type)
        extract_multiple_segments_to_single_file(file_path_transcript, segment_numbers,  file_path_episode,
                                                 "segments", session_id)
        time.sleep(random.randint(2, 5))


st.set_page_config(
    page_title="PodSum.app",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.podsum.app/help',
        'Report a bug': "https://www.podsum.app/bug",
        'About': "# Create an audio summary from your newest podcast episode!"
    }
)




left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('PodSumAppBeta.png', width=200)

with st.form("my_form"):
   st.write("**This is how it works:**")

   multi = '''Welcome to PodSum.app. Your AI tool for podcast audio summaries. Upload your podcast episode to be summarized, your intro and a separator sound. Then click on the "Sum it!" button and wait until the summary has been created. The summary can be downloaded as a MP3-file.
   '''

   st.markdown(multi)

   st.divider()

   # Upload only mp3 files with max 20mb size
   uploaded_file_episode = st.file_uploader("Upload Podcast Episode", type=['mp3'],
                                    help="Lade hier die MP3 deiner Podcast-Episode hoch, für die eine Audio-Zusammenfassung erstellt werden soll.",
                                    accept_multiple_files=False)


   # Upload only mp3 files with max 20mb size
   uploaded_file_intro = st.file_uploader("Upload Summary-Intro", type=['mp3'],
                                    help="Lade hier dein Intro für die Audio-Zusammenfassung hoch. Zum Beispiel: Willkommen zu meinem Podcast XY. In der heutigen Episode wird es um XY gehen und hören wir jetzt mal rein...",
                                    accept_multiple_files=False)


   # Upload only mp3 files with max 20mb size
   uploaded_file_separator = st.file_uploader("Upload Segment Separator", type=['mp3'],
                                    help="Der Segment Separator wird verwendet, um die einzelnen Impressionen aus der Podcast-Episode voneinander abzugrenzen.",
                                    accept_multiple_files=False)

   submitted = st.form_submit_button("Sum it!")

   # Generierung einer fünfstelligen Zufallszahl
   session_id = random.randint(10000, 99999)


   if submitted:
       with st.spinner('Processing has been started (this may take a while)...'):
           if uploaded_file_episode is not None and uploaded_file_intro is not None and uploaded_file_separator is not None:

               file_path_transcript = os.path.join("transcript",
                                                   f"{session_id}_transcript_{uploaded_file_episode.name}.txt")
               file_path_export = os.path.join("export", f"{session_id}_podSummarized_{uploaded_file_episode.name}")
               file_path_episode = os.path.join("episode", f"{session_id}_{uploaded_file_episode.name}")
               file_path_intro = os.path.join("intro", f"{session_id}_{uploaded_file_intro.name}")
               file_path_separator = os.path.join("separator", f"{session_id}_{uploaded_file_separator.name}")

               with open(file_path_episode, "wb") as f:
                   f.write(uploaded_file_episode.getbuffer())
               with open(file_path_intro, "wb") as f:
                   f.write(uploaded_file_intro.getbuffer())
               with open(file_path_separator, "wb") as f:
                   f.write(uploaded_file_separator.getbuffer())

               transcribe_podcast(file_path_episode, file_path_transcript)
               st.success("Listened to your podcast episode and transcribed it!")

               extracted_text = extract_max_1000_words(file_path_transcript)
               topic, type = get_type_and_topic(extracted_text)

               process_transcription_in_blocks(file_path_transcript, file_path_episode, session_id, topic, type)
               st.success("Found the most relevant segments!")
               merge_mp3_with_separator("segments", file_path_separator, file_path_export, session_id, file_path_intro)
               st.success("Merged the segments and created the summary!")
               if file_path_export is not None and check_mp3_integrity(file_path_export):
                   st.write("Listen to your summary and download it!")
                   delete_files_with_number('segments', session_id)
                   delete_files_with_number('episode', session_id)
                   delete_files_with_number('intro', session_id)
                   delete_files_with_number('separator', session_id)
                   delete_files_with_number('transcript', session_id)
                   audio_file = open(file_path_export, 'rb')
                   audio_bytes = audio_file.read()
                   st.audio(audio_bytes, format='audio/mp3')


               else:
                   st.write("Da ist etwas schief gegangen. Probiere es bitte erneut")
           else:
               st.write("Es wurden keine Podcastfolge, Intro oder Separator hochgeladen")
       st.success('Done!')



st.write("**All of your data will be deleted after summirization.**")
st.write("Impressum/Datenschutz")


