import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "RAG Academy",
  description: "Learn every RAG pattern from Basic to Production — with live traces",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} h-full`}>
      <body className="min-h-full bg-gray-950 text-gray-100 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
