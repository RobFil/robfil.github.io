-- Run this once for projects created before the explicit Data API grants.
-- Row Level Security policies still restrict rows to the member's household.
grant usage on schema public to authenticated;
grant select on table public.households to authenticated;
grant select, insert, update on table public.products to authenticated;
grant select, insert on table public.inventory_events to authenticated;
