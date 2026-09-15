import Link from "next/link";

import { Button } from "@/components/ui/button";

/**
 * The 404.
 *
 * It links to the pages that exist rather than only offering "go home",
 * because the most common way to land here is a stale link from an earlier
 * version of the navigation.
 */
export default function NotFound() {
  return (
    <main className="grid min-h-screen place-items-center bg-background px-6">
      <div className="flex max-w-md flex-col items-center text-center">
        <p className="font-mono text-2xs uppercase tracking-widest text-muted-foreground">
          404
        </p>
        <h1 className="mt-2 text-xl font-medium tracking-tight">
          That page does not exist
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          The link may be from an older version of the app, or the id in the URL
          may belong to another workspace.
        </p>

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <Button asChild size="sm">
            <Link href="/dashboard">Dashboard</Link>
          </Button>
          <Button asChild size="sm" variant="outline">
            <Link href="/documents">Documents</Link>
          </Button>
          <Button asChild size="sm" variant="outline">
            <Link href="/runs">Runs</Link>
          </Button>
          <Button asChild size="sm" variant="ghost">
            <Link href="/help">Help</Link>
          </Button>
        </div>
      </div>
    </main>
  );
}
