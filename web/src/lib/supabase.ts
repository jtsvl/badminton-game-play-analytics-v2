import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL ?? ''
const anon = import.meta.env.VITE_SUPABASE_ANON_KEY ?? ''

/** True when the Supabase env is configured (lets the UI show a helpful message otherwise). */
export const isConfigured = Boolean(url && anon)

export const supabase = createClient(url, anon)
