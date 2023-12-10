
import pysrt

from openai import OpenAI
client = OpenAI()

audio_file = open("video-English-en-US.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="srt"
)


print(transcript)

srt = pysrt.from_string(transcript)

srt.save('./test.srt')

print(srt[0].start, srt[0].end, srt[0].text, srt[0].duration.ordinal)
