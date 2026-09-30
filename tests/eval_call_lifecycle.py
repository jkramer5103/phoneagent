"""Optional live Responses API regressions. No calls or generated audio.

Run from the project directory: python3 tests/eval_call_lifecycle.py
Requires OPENAI_API_KEY in the environment or .env; incurs API usage.
"""
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from call_lifecycle import review_lifecycle
from telephone_agent import load_env_value, calendar_context
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
def run(case):
 name,task,pending,history,expected=case
 started=time.monotonic()
 decision=review_lifecycle(key,'gpt-6-luna',{'task':task,'calendar':calendar_context(),'history':history,'native_closing_pending':pending})
 print(name,round(time.monotonic()-started,2),decision,flush=True)
 assert decision.action==expected,(name,decision.action,expected)
 if name in {'actual-call-overlooked-cheaper-option', 'generic-overlooked-alternative'}:
  assert decision.message.strip(), (name, 'Missing clarification')
with ThreadPoolExecutor(max_workers=2) as pool:
 list(pool.map(run,cases))
print('All semantic lifecycle cases passed.')
