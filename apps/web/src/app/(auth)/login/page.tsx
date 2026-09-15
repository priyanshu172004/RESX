import { Suspense } from "react";
import type { Metadata } from "next";

import { AuthForm } from "@/components/auth-form";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  // `useSearchParams` (for the ?next= redirect) requires a Suspense boundary
  // during prerender, otherwise the build fails rather than the page.
  return (
    <Suspense>
      <AuthForm mode="login" />
    </Suspense>
  );
}
