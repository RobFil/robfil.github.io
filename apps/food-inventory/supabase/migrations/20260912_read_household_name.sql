-- Run this once if the initial schema was already applied before this file existed.
-- RLS continues to limit visible households to their members.
grant select on table public.households to authenticated;
