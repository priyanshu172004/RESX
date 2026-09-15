import type { Metadata } from "next";

import SaasLanding from "@/components/ui/saas-template";

export const metadata: Metadata = {
  title: "RESX — every number cited, every number computed",
  description:
    "Upload a corpus of business documents, ask a question, and get an executive report where every figure traces to a citation and to a logged computation.",
  // The landing page is the one surface that should be indexable; the root
  // layout opts the whole app out, so this overrides it here rather than
  // flipping the default and having to remember to opt every private page out.
  robots: { index: true, follow: true },
};

export default function LandingPage() {
  return <SaasLanding />;
}
