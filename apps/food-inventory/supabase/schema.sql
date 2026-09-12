-- Run this once in the Supabase SQL Editor. It creates a household-scoped,
-- event-based sync backend. The browser uses only the public anon key.

create table if not exists public.households (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(trim(name)) between 1 and 80),
  invite_code text not null unique,
  created_at timestamptz not null default now()
);

create table if not exists public.household_members (
  household_id uuid not null references public.households(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  joined_at timestamptz not null default now(),
  primary key (household_id, user_id)
);

create table if not exists public.products (
  id text primary key,
  household_id uuid not null references public.households(id) on delete cascade,
  barcode text,
  name text not null,
  generic_ingredient text,
  source text not null check (source in ('manual', 'open_food_facts')),
  created_at timestamptz not null,
  updated_at timestamptz not null
);
create unique index if not exists products_household_barcode_unique
  on public.products (household_id, barcode) where barcode is not null;

create table if not exists public.inventory_events (
  id uuid primary key,
  household_id uuid not null references public.households(id) on delete cascade,
  product_id text not null references public.products(id) on delete restrict,
  event_type text not null check (event_type in ('purchase', 'consume', 'correction', 'manual_add', 'manual_remove')),
  quantity_change numeric not null,
  timestamp timestamptz not null,
  device_id uuid not null
);
create index if not exists inventory_events_household_id on public.inventory_events (household_id);

alter table public.households enable row level security;
alter table public.household_members enable row level security;
alter table public.products enable row level security;
alter table public.inventory_events enable row level security;

create or replace function public.is_household_member(target_household_id uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.household_members
    where household_id = target_household_id and user_id = auth.uid()
  );
$$;

create policy "Members can see their memberships" on public.household_members
  for select using (user_id = auth.uid());
create policy "Members can see their households" on public.households
  for select using (public.is_household_member(id));
create policy "Members can read products" on public.products
  for select using (public.is_household_member(household_id));
create policy "Members can create products" on public.products
  for insert with check (public.is_household_member(household_id));
create policy "Members can update products" on public.products
  for update using (public.is_household_member(household_id)) with check (public.is_household_member(household_id));
create policy "Members can read events" on public.inventory_events
  for select using (public.is_household_member(household_id));
create policy "Members can create events" on public.inventory_events
  for insert with check (public.is_household_member(household_id));

create or replace function public.create_household(household_name text)
returns table (household_id uuid, invite_code text)
language plpgsql security definer set search_path = public as $$
declare created_household_id uuid; created_invite_code text;
begin
  if auth.uid() is null then raise exception 'Not authenticated'; end if;
  created_invite_code := upper(left(replace(gen_random_uuid()::text, '-', ''), 12));
  insert into public.households (name, invite_code) values (trim(household_name), created_invite_code)
    returning id into created_household_id;
  insert into public.household_members (household_id, user_id) values (created_household_id, auth.uid());
  return query select created_household_id, created_invite_code;
end;
$$;

create or replace function public.join_household(household_invite_code text)
returns uuid language plpgsql security definer set search_path = public as $$
declare target_household_id uuid;
begin
  if auth.uid() is null then raise exception 'Not authenticated'; end if;
  select id into target_household_id from public.households where invite_code = upper(trim(household_invite_code));
  if target_household_id is null then raise exception 'Invalid invitation code'; end if;
  insert into public.household_members (household_id, user_id) values (target_household_id, auth.uid()) on conflict do nothing;
  return target_household_id;
end;
$$;

grant execute on function public.create_household(text) to authenticated;
grant execute on function public.join_household(text) to authenticated;

-- Data API access is explicit. RLS policies above still decide which household
-- rows an authenticated user may see or change.
revoke all on table public.households, public.household_members, public.products, public.inventory_events from anon;
grant usage on schema public to authenticated;
grant select on table public.households to authenticated;
grant select, insert, update on table public.products to authenticated;
grant select, insert on table public.inventory_events to authenticated;

alter publication supabase_realtime add table public.products, public.inventory_events;
