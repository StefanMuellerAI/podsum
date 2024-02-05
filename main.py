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
from mutagen.mp3 import MP3
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel


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


def transcribe_podcast_faster(file_path, output_file):
    model_size = "tiny"
    # Passen Sie die device und compute_type Parameter entsprechend Ihrer Umgebung an
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    i = 0
    segments, info = model.transcribe(file_path, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    with open(output_file, "w") as f:
        for segment in segments:  # Verwenden Sie segments direkt ohne enumerate
           i = i + 1
           f.write(f"Segment {i}: {segment.start}-{segment.end}: {segment.text}\n")





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
                faded_file
            ])
            faded_files.append(faded_file)

    return faded_files


def merge_mp3_with_separator(segments_folder, file_path_separator, output_file, session_id, toggle_intro, toggle_fade,
                             file_path_intro=None):
    # Löschen der spezifischen Datei, falls vorhanden
    specific_file_to_delete = os.path.join(segments_folder, f"0_{session_id}_output_segment.mp3")
    if os.path.exists(specific_file_to_delete):
        os.remove(specific_file_to_delete)

    # Anwenden von Fade-In und Fade-Out auf jedes Segment
    if toggle_fade:
        segment_files = apply_fade_to_segments(segments_folder, session_id)
    else:
        segment_files = glob.glob(os.path.join(segments_folder, '*.mp3'))

    # Segmente nach der ersten Zahl im Dateinamen sortieren
    segment_files_sorted = sorted(segment_files, key=lambda x: int(os.path.basename(x).split('_')[0]))

    # Erstellen einer temporären Datei, die das Intro, alle Segmente und Trenner kombiniert
    with open(f"{session_id}_temp_filelist.txt", "w") as filelist:
        # Zuerst die Intro-Datei hinzufügen
        if toggle_intro and os.path.exists(file_path_intro):
            filelist.write(f"file '{file_path_intro}'\n")

        # Einfügen des Trenners nur zwischen Segmenten, nicht vor dem ersten Segment
        for i, segment_file in enumerate(segment_files_sorted):
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
        output_file
    ])

    # Löschen der temporären Dateiliste und der gefadeten Dateien, wenn aktiviert
    os.remove(f"{session_id}_temp_filelist.txt")
    if toggle_fade:
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
             "content": f"Der Podcast aus dem die folgenden Segmente stammen hat folgendes Thema: {topic}."
                        f"Identifiziere bitte jeweils mindestens 3 oder mehr (weniger als 12) zusammenhängende, numerisch mit jeweils + 1 aufeinanderfolgende Segmente (es darf keine Segmente geben, die mehr als 1 auseinanderliegen), die eine für den Zuhörenden extrem relevante Aussage wiederspiegeln. Also etwas, was defintiv in eine Audio-Zusammenfassung hineingehören würde. Achte darauf die Segmente so auszuwählen, dass die Kernaussage komplett enthalten ist. "
                        f"Es dürfen keine Eigenwerbung, Verweise auf Websites oder ähnliches enthalten sein."
                        f"Es existieren 3 verschiedene Podcast Typen: Das sind 1. Interviews, 2. Talkrunden und 3. Solos. Die folgenden Segmente stammen aus einem Podcast vom Typ: {type}."
                        f"Wenn der Type 'Solo' ist, beachte bitte alle Inhalte. Wenn der Typ 'Interview' ist, beachte bitte nur die Inhalte des antwortenden Gastes. Wenn der Typ 'talk' ist, achte auf alle Aussagen."
                        f"Gib ausschließlich die Segmentnummern per Komata getrennt zurück ohne Leerzeichen. Wenn du keine relevanten Segmente mehr findest, gib 0 aus. Hier ist sind die transkribierten Segmente: {block}"}
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
    output_file = os.path.join(output_dir, f"{segment_numbers_str}_{session_id}_output_segment.mp3")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_file,
        "-af", f"aselect='{filter_string}',asetpts=N/SR/TB",
        "-acodec", "libmp3lame",
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

def process_transcription_in_blocks(file_path_transcript, file_path_episode, session_id, topic, type, selected_length):
    with open(file_path_transcript, "r") as file:
        segments = file.readlines()
    print(len(segments))
    print(selected_length)

    block_size = calculate_blocks(file_path_episode, selected_length, len(segments))

    for i in range(0, len(segments), block_size):
        block = segments[i:i + block_size]
        segment_numbers = select_segments(block, topic, type)
        extract_multiple_segments_to_single_file(file_path_transcript, segment_numbers,  file_path_episode,
                                                 "segments", session_id)
        time.sleep(random.randint(2, 5))

def calculate_blocks(file_path_episode, selected_length, total_segments):

    total_length = get_mp3_length(file_path_episode)
    segment_length = 13.5

    if selected_length == "Short":
        wished_length = total_length / 20
    elif selected_length == "Middle":
        wished_length = total_length / 10
    elif selected_length == "Long":
        wished_length = total_length / 5
    else:
        raise ValueError("Unbekanntes Wort. Bitte wähle 'small', 'middle' oder 'long'.")

    necessary_runs = wished_length / segment_length
    blocks = total_segments / necessary_runs
    return round(blocks)

def get_mp3_length(file_path_episode):
    audio = MP3(file_path_episode)
    return audio.info.length

def get_session_id():
    # Generierung einer fünfstelligen Zufallszahl
    session_id = random.randint(10000, 99999)
    return session_id

session_id = get_session_id()

# Streamlit App
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

left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('PodSumAppBeta.png', width=200)


st.write("**This is how it works:**")

multi = '''Welcome to PodSum.app v0.2. Your AI tool for podcast audio summaries. This tool is under constant development and maybe there are some bugs to be figured out ;) Upload your podcast episode to be summarized, add your intro and a separator sound. Then click on the "Sum it!" and wait until the summary has been created. After that the summary can be downloaded as a MP3-file. PodSum analyzes your episode for type and topic and cuts relevant passages into an audio summary. Further functions will follow. 
'''
st.markdown(multi)

st.divider()

left_co2, last_co2 = st.columns(2)
with left_co2:
    selected_length = st.radio(
        "How long should your summary be?",
        ["Short", "Middle", "Long"],
        index=1,)

with last_co2:
    toggle_fade = st.toggle('Need Segment Fade-In?', value=True)

st.divider()


# Upload only mp3 files with max 30mb size
uploaded_file_episode = st.file_uploader("Upload Your Podcast Episode :red[*Mandatory]", type=['mp3'],
                                    help="Upload the MP3 of your podcast episode for which you want to create an audio summary here.",
                                    accept_multiple_files=False)

# Upload only mp3 files with max 30mb size
uploaded_file_separator = st.file_uploader("Upload Your Segment Separator  :red[*Mandatory]", type=['mp3'],
                                    help="The segment separator is used to separate the individual impressions from the podcast episode.",
                                    accept_multiple_files=False)

# Upload only mp3 files with max 30mb size
uploaded_file_intro = st.file_uploader("Upload Your Summary-Intro :green[*Optional]", type=['mp3'],
                                    help="Upload your intro for the audio summary here. For example: Welcome to my podcast XY. Today's episode will be about XY and let's listen in now...",
                                    accept_multiple_files=False)

if st.button('Sum it!', type="primary"):
    with st.spinner("Summarizing your podcast..."):
        if uploaded_file_episode is not None and uploaded_file_separator is not None:
            file_path_transcript = os.path.join("transcript", f"{session_id}_transcript_{uploaded_file_episode.name}.txt")
            file_path_export = os.path.join("export", f"{session_id}_podSummarized_{uploaded_file_episode.name}")
            file_path_episode = os.path.join("episode", f"{session_id}_{uploaded_file_episode.name}")
            file_path_separator = os.path.join("separator", f"{session_id}_{uploaded_file_separator.name}")

            with open(file_path_episode, "wb") as f:
                f.write(uploaded_file_episode.getbuffer())

            with open(file_path_separator, "wb") as f:
                f.write(uploaded_file_separator.getbuffer())

            # Old Transcription
            # transcribe_podcast(file_path_episode, file_path_transcript)

            # New Transcription
            transcribe_podcast_faster(file_path_episode, file_path_transcript)
            st.success("Listened to your podcast episode and transcribed it!")

            extracted_text = extract_max_1000_words(file_path_transcript)
            topic, type = get_type_and_topic(extracted_text)

            process_transcription_in_blocks(file_path_transcript, file_path_episode, session_id, topic, type, selected_length)
            st.success("Found the most relevant segments!")

            if uploaded_file_intro is not None:
                toggle_intro = True
                file_path_intro = os.path.join("intro", f"{session_id}_{uploaded_file_intro.name}")
                with open(file_path_intro, "wb") as f:
                    f.write(uploaded_file_intro.getbuffer())
                merge_mp3_with_separator("segments", file_path_separator, file_path_export, session_id,
                                         toggle_intro, toggle_fade, file_path_intro)
            else:
                toggle_intro = False
                merge_mp3_with_separator("segments", file_path_separator, file_path_export, session_id, toggle_intro, toggle_fade)


            st.success("Merged the segments and created the summary!")

            if file_path_export is not None and check_mp3_integrity(file_path_export):
                audio_file = open(f"{file_path_export}", 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')

                if st.download_button(label="Download your summary", data=audio_bytes, file_name=f"{file_path_export}", mime="audio/mp3"):
                    st.rerun()

                delete_files_with_number('segments', session_id)
                delete_files_with_number('episode', session_id)
                delete_files_with_number('intro', session_id)
                delete_files_with_number('separator', session_id)
                delete_files_with_number('transcript', session_id)
                #delete_files_with_number('export', session_id)

            else:
                st.write("Something went wrong. Please try again.")
        else:
            st.write("There are files missing. Please upload all files.")

st.divider()

st.write("**All of your data will be deleted after summirization.**")
st.write("[Impressum](https://stefanai.de/impressum)/[Datenschutz](https://stefanai.de/datenschutz)")


