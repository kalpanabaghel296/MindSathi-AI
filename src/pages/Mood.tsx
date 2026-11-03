import { useState } from 'react'
import { useMood } from '../store/mood'

const moods = ['Low', 'Down', 'Okay', 'Good', 'Great']

export default function Mood() {
  const { entries, add, clear } = useMood()
  const [v, setV] = useState(0)
  const [note, setNote] = useState('')
  const [msg, setMsg] = useState('')
  function save() {
    if (!v) return setMsg('Pick a mood first.')
    add({ v, note: note.trim().slice(0, 200), date: new Date().toLocaleDateString('en-IN') })
    setV(0); setNote(''); setMsg('Check-in saved.')
  }
  return (
    <div>
      <h1 className="m-0 text-3xl font-extrabold">Mood tracking</h1>
      <p className="mt-1 mb-6 text-mute">A calm, simple daily check-in.</p>
      <section className="rounded-3xl border border-line bg-white p-6">
        <div className="grid grid-cols-5 gap-2" role="group" aria-label="Mood">
          {moods.map((m, i) => (
            <button key={m} type="button" aria-pressed={v === i + 1} onClick={() => setV(i + 1)} className={`min-h-16 rounded-2xl border-2 font-semibold ${v === i + 1 ? 'border-teal bg-mint' : 'border-line bg-white'}`}>{m}</button>
          ))}
        </div>
        <label className="mt-4 block text-sm font-semibold" htmlFor="note">Journal (optional)</label>
        <textarea id="note" value={note} onChange={(e) => setNote(e.target.value)} rows={3} className="mt-1.5 w-full rounded-xl border border-line p-3" />
        <div className="mt-3 flex items-center gap-3">
          <button onClick={save} className="min-h-11 rounded-xl bg-forest px-5 font-semibold text-white">Save check-in</button>
          <button onClick={clear} className="min-h-11 rounded-xl border border-forest px-5 font-semibold text-forest">Clear history</button>
          <span role="status" className="text-sm text-mute">{msg}</span>
        </div>
      </section>
      <section className="mt-4 rounded-3xl border border-line bg-white p-6">
        <p className="m-0 mb-2 text-xs font-bold tracking-widest text-mute uppercase">Recent check-ins</p>
        {entries.length === 0 ? <p className="text-mute">No check-ins yet.</p> : [...entries].reverse().slice(0, 7).map((e, i) => (
          <div key={i} className="flex justify-between gap-3 border-b border-line py-2 last:border-0"><span>{e.date} · {moods[e.v - 1]}</span><span className="text-sm text-mute">{e.note || 'No note'}</span></div>
        ))}
      </section>
    </div>
  )
}
