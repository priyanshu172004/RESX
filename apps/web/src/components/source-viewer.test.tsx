/**
 * The citation viewer, and the one thing it must never do quietly.
 *
 * A quote that cannot be found on the page it cites is the most important
 * thing this panel can tell a reader. Rendering the page without comment in
 * that case would imply the quote had been checked — which is worse than not
 * showing the page at all, because it looks like verification.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SourceViewer } from "./source-viewer";
import type { Citation } from "@/lib/api";

const page = vi.fn();
vi.mock("@/lib/api", () => ({
  documents: { page: (...args: unknown[]) => page(...args) },
}));

function citation(overrides: Partial<Citation> = {}): Citation {
  return {
    citation_id: "cit_1",
    doc_id: "doc_abc",
    page: 4,
    para_idx: 2,
    quote: "Revenue rose 16.7% year on year",
    resolution: "ok",
    match_score: 0.97,
    ...overrides,
  };
}

beforeEach(() => {
  page.mockReset();
});

describe("SourceViewer", () => {
  it("fetches nothing until it is opened", () => {
    render(
      <SourceViewer citation={citation()} open={false} onOpenChange={() => {}} />,
    );
    expect(page).not.toHaveBeenCalled();
  });

  it("loads the cited page when opened", async () => {
    page.mockResolvedValue({
      doc_id: "doc_abc",
      page: 4,
      text: "Revenue rose 16.7% year on year, driven by volume.",
    });

    render(<SourceViewer citation={citation()} open onOpenChange={() => {}} />);

    await waitFor(() => expect(page).toHaveBeenCalledWith("doc_abc", 4));
  });

  it("highlights the quote inside the page", async () => {
    page.mockResolvedValue({
      doc_id: "doc_abc",
      page: 4,
      text: "Preamble. Revenue rose 16.7% year on year. Footnote.",
    });

    render(<SourceViewer citation={citation()} open onOpenChange={() => {}} />);

    const mark = await screen.findByText("Revenue rose 16.7% year on year");
    expect(mark.tagName).toBe("MARK");
  });

  it("finds a quote whose separators differ from the page", async () => {
    // Table chunks are pipe-joined and page text is space-separated, so a
    // faithful quote routinely fails a literal search against its own page.
    page.mockResolvedValue({
      doc_id: "doc_abc",
      page: 4,
      text: "North 291.6 33.8",
    });

    render(
      <SourceViewer
        citation={citation({ quote: "North | 291.6 | 33.8" })}
        open
        onOpenChange={() => {}}
      />,
    );

    await waitFor(() =>
      expect(document.querySelector("mark")).not.toBeNull(),
    );
  });

  it("says so when the quote is not on the page", async () => {
    page.mockResolvedValue({
      doc_id: "doc_abc",
      page: 4,
      text: "Something else entirely.",
    });

    render(<SourceViewer citation={citation()} open onOpenChange={() => {}} />);

    expect(
      await screen.findByText(/was not found on this page/i),
    ).toBeDefined();
  });

  it("still shows the page when the quote is missing", async () => {
    // So the reader can judge it rather than being told nothing.
    page.mockResolvedValue({
      doc_id: "doc_abc",
      page: 4,
      text: "Something else entirely.",
    });

    render(<SourceViewer citation={citation()} open onOpenChange={() => {}} />);

    expect(await screen.findByText(/Something else entirely/)).toBeDefined();
  });

  it("offers the link instead for a web citation", async () => {
    render(
      <SourceViewer
        citation={citation({
          doc_id: null,
          page: null,
          url: "https://example.com/report",
        })}
        open
        onOpenChange={() => {}}
      />,
    );

    const link = await screen.findByRole("link", { name: /open the source/i });
    expect(link.getAttribute("href")).toBe("https://example.com/report");
    // A web citation has no extracted text, so nothing should be fetched.
    expect(page).not.toHaveBeenCalled();
  });

  it("reports a failed load rather than showing an empty panel", async () => {
    page.mockRejectedValue(new Error("no such page"));

    render(<SourceViewer citation={citation()} open onOpenChange={() => {}} />);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("no such page");
  });
});
