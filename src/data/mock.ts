// Illustrative demo values. Replace by backend responses via src/services.
export const THRESHOLDS = { moderate: 0.2727, high: 0.4227 } // from the project model, not clinical cut-offs
export const riskHistory = [0.21, 0.24, 0.22, 0.3, 0.33, 0.31, 0.38, 0.41]
export const sleepWeek = [6.8, 7.1, 6.2, 5.9, 7.4, 6.0, 6.5].map((v, i) => ({ day: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i], hours: v }))
// Global SHAP (mean |SHAP|) from the project's LightGBM run on test participants.
export const shapGlobal = [
  { name: 'money', value: 0.3, context: true }, { name: 'comorbid migraines', value: 0.227, context: true },
  { name: 'comorbid neuropathic', value: 0.222, context: true }, { name: 'migraine days', value: 0.133, context: true },
  { name: 'birth year', value: 0.106, context: true }, { name: 'steps rolling sum', value: 0.075, context: false },
  { name: 'sleep weekend asleep mean', value: 0.056, context: false }, { name: 'sleep nap count', value: 0.055, context: false },
]
