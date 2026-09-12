interface PlaceholderPageProps {
  title: string;
  description: string;
}

export function PlaceholderPage({ title, description }: PlaceholderPageProps) {
  return (
    <section className="page placeholder-page">
      <header className="page-header">
        <h1>{title}</h1>
      </header>
      <p>{description}</p>
    </section>
  );
}
