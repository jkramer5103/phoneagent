"""Optional live Responses API regressions. No calls or generated audio.

Run from the project directory: python3 tests/eval_call_lifecycle.py
Requires OPENAI_API_KEY in the environment or .env; incurs API usage.
"""
import sys, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from call_lifecycle import review_lifecycle
from telephone_agent import load_env_value, calendar_context
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', default='gpt-6-sol')
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()
key=load_env_value('OPENAI_API_KEY')
if not key:
 raise SystemExit('Missing OPENAI_API_KEY')
def message(speaker,text):return {'speaker':speaker,'text':text}
cases=[
 ('new-offer-before-unspoken-farewell','Buche einen Reifenwechsel bis 60 Euro auf Kramer.',True,[
  message('other_person','40 Euro Reifen und 30 Euro Montage.'),
  message('agent','Das liegt über meinem Budget. Dann buche ich keinen Termin. Vielen Dank.'),
  message('other_person','Was ist Ihr Budget?'),
  message('agent','Maximal 60 Euro.'),
  message('other_person','Ein Standardreifen kostet 30 Euro plus 30 Montage, also insgesamt 60 Euro. Morgen um 16 Uhr wäre frei.')], 'keep'),
 ('failed-call-already-said-goodbye','Buche einen Reifenwechsel morgen zwischen 14 und 17 Uhr bis 60 Euro auf Kramer.',False,[
  message('agent','Welche Termine hätten Sie morgen zwischen 14 und 17 Uhr?'),
  message('other_person','Das geht nur, wenn Sie mir die Fahrradmarke nennen.'),
  message('agent','Die Marke kenne ich nicht. Geht es trotzdem?'),
  message('other_person','Nein, ohne Marke geht es nicht.'),
  message('agent','Okay, vielen Dank erstmal. Das reicht mir für den Moment. Auf Wiederhören.')], 'end'),
 ('native-close-before-any-farewell','Buche einen Reifenwechsel bis 60 Euro. Falls es teurer ist, beende ohne Buchung.',True,[
  message('agent','Was kostet Reifen und Montage zusammen?'),
  message('other_person','40 Euro für den Reifen und 30 Euro Montage. Passt das?')], 'say_goodbye'),
 ('ordinary-offer-not-yet-confirmed','Buche einen Reparaturtermin bis 60 Euro auf Kramer.',False,[
  message('agent','Was kostet die Reparatur?'),
  message('other_person','50 Euro, morgen um 15 Uhr wäre frei.')], 'keep'),
 ('quoted-farewell-not-a-real-ending','Frag nach den Öffnungszeiten.',False,[
  message('other_person','Meine Kollegin hat vorhin Auf Wiederhören gesagt, aber ich bin noch hier. Welche Öffnungszeiten interessieren Sie?'),
  message('agent','Wie lange haben Sie am Freitag geöffnet?')], 'keep'),
 ('caller-explicitly-stops-unfinished-call','Frag wann das Paket abgeholt werden kann.',False,[
  message('agent','Wann kann ich es abholen?'),
  message('other_person','Bitte legen Sie jetzt auf.')], 'end'),
]
cases += [
 ('actual-call-overlooked-cheaper-option', 'Buche einen Reifenwechsel morgen 14-17 Uhr bis 60 Euro auf Kramer. Wenn es teurer ist, beende ohne Buchung.', True, [
 message('agent','Hätten Sie morgen zwischen 14 und 17 Uhr einen Termin?'),
 message('other_person','Um 16:25 hätten wir etwas. 40 Euro für unseren besonders pannensicheren Reifen. Es würde auch günstiger gehen, aber den empfehlen wir sehr. Und 30 Euro für die Montage.'),
 message('agent','Das wären insgesamt 70 Euro. Das liegt über meinem Rahmen. Trotzdem vielen Dank für Ihre Zeit. Auf Wiederhören!'),
 message('other_person','Okay, ja, dann können')], 'keep'),
 ('generic-overlooked-alternative', 'Buche eine Übernachtung morgen bis 90 Euro auf Kramer.', True, [
 message('other_person','Das Zimmer mit Aussicht kostet 110 Euro. Wir hätten auch ein günstigeres Zimmer ohne Aussicht.'),
 message('agent','110 Euro ist zu teuer. Danke, auf Wiederhören.')], 'keep'),
 ('valid-farewell-followed-by-hello','Frage nach den Öffnungszeiten am Freitag.',False,[
 message('agent','Wann haben Sie am Freitag geöffnet?'),
 message('other_person','Von 10 bis 18 Uhr.'),
 message('agent','Vielen Dank, auf Wiederhören.'),
 message('other_person','Hallo?')], 'end'),
 ('firm-refusal-no-invented-alternative','Buche eine Übernachtung morgen bis 90 Euro.',False,[
 message('other_person','Unser einziges freies Zimmer kostet 110 Euro. Günstiger geht es nicht.'),
 message('agent','Das liegt über meinem Rahmen. Vielen Dank, auf Wiederhören.')], 'end'),
]
# Preserve the imperfect transcript, rather than rewriting it into clean speech.
full_task = """Du rufst bei einer Fahrradwerkstatt an. Frage nach einem Termin für einen
Reifenwechsel am Hinterrad eines normalen Citybikes, morgen
zwischen 14 und 17 Uhr. Nimm nur einen Termin in diesem Zeitfenster an.
Frage auch nach dem ungefähren Preis inklusive Reifen
und Montage. Bis 60 Euro darfst du den Termin unter dem Namen Kramer annehmen.
Falls morgen nichts frei ist, frage nach einem Termin übermorgen im gleichen
Zeitfenster. Warte auf die Terminbestätigung. Ist es teurer oder kein passender
Termin frei, bedanke dich und beende das Gespräch ohne Buchung. Sprich kurz
und freundlich auf Deutsch; erfinde keine weiteren Angaben."""
last_offer = ('Ja, das ist nicht ganz ideal, aber sollte passen. Wir hätten für unsere '
              'neuen Extraplatz für den Reifen ein Sonderangebot. 40 Euro für den Reifen, '
              'ist nicht ganz günstig. Das lohnt sich aber. Haben Sie dann auch die '
              'günstigere aber würde eigentlich platt- äh plattfesten empfehlen. '
              'Und dann noch 30 Euro für die Montage. Passt das für Sie? '
              'Äh 16 Uhr übrigens der Termin.')
for ending in ('Einen Moment, ich rechne kurz.',
               'Danke für die Auskunft. Dann buche ich den Termin nicht. Auf Wiederhören.'):
 cases.append(('verbatim-failed-call-' + ending[:5], full_task, True, [
  message('agent','Kann man auch ohne die Angabe einen Termin dafür bekommen?'),
  message('other_person',last_offer), message('agent',ending)], 'keep'))

def run(case):
 name,task,pending,history,expected=case
 started=time.monotonic()
 decision=review_lifecycle(key,args.model,{'task':task,'calendar':calendar_context(),'history':history,'native_closing_pending':pending})
 print(name,round(time.monotonic()-started,2),decision,flush=True)
 assert decision.action==expected,(name,decision.action,expected)
 if name in {'actual-call-overlooked-cheaper-option', 'generic-overlooked-alternative'} or name.startswith('verbatim-failed-call'):
  assert decision.message.strip(), (name, 'Missing clarification')
with ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(run,cases * args.repeat))
print('All semantic lifecycle cases passed.')
