type FindingCardProps = {
  detail: string;
  label: string;
  value: string;
};

export function FindingCard({ detail, label, value }: FindingCardProps) {
  return (
    <article className="panel finding-card">
      <div className="eyebrow">{label}</div>
      <strong>{value}</strong>
      <p>{detail}</p>
    </article>
  );
}
