import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://jyuxqhkutlbnzrutuey.supabase.co'
const supabaseAnonKey = 'YOUR_SUPABASE_ANON_KEY' // Found in your Project Settings

export const supabase = createClient(supabaseUrl, supabaseAnonKey)