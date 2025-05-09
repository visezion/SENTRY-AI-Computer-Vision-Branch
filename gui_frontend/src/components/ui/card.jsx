export function Card({ children, className = "" }) {
    return <div className={`bg-white shadow rounded-xl p-4 ${className}`}>{children}</div>;
  }
  
  export function CardContent({ children }) {
    return <div className="text-gray-700">{children}</div>;
  }
  