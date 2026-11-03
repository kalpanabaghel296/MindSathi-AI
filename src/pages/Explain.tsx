import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { explainService } from '../services/api'

type Row = { name: string; value: number; context: boolean }

export default function Explain() {
  const [rows, setRows] = useState<Row[]>([])
  useEffect(() => { explainService.global().then(setRows) }, [])
  return (
    <div>
      <h1 className="m-0 text-3xl font-extrabold">Explainable AI</h1>
      <p className="mt-1 mb-6 text-mute">Which factors shaped the model’s estimates (global SHAP).</p>
      <section className="rounded-3xl border border-line bg-white p-6">
        <div className="h-96"><ResponsiveContainer><BarChart data={rows} layout="vertical" margin={{ left: 40 }}>
          <XAxis type="number" /><YAxis type="category" dataKey="name" width={170} /><Tooltip />
          <Bar dataKey="value" name="mean |SHAP|" radius={[0, 8, 8, 0]}>{rows.map((r) => <Cell key={r.name} fill={r.context ? '#D9A23B' : '#2A6F6B'} />)}</Bar>
        </BarChart></ResponsiveContainer></div>
        <p className="text-sm text-mute">Amber: context features (demographic, comorbidity). Teal: behavior (steps, sleep). Values come from this project’s LightGBM run on test participants. SHAP describes what the model used, not what causes depression.</p>
      </section>
    </div>
  )
}
