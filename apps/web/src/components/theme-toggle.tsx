"use client";

import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";

/**
 * Theme toggle with no mount state and no effect.
 *
 * The usual next-themes pattern gates rendering on a `mounted` flag set inside
 * an effect, which costs a cascading render and is rejected by React Compiler.
 * Both icons are rendered instead and CSS picks the right one from the `.dark`
 * class already on `<html>` — so the correct icon is present in the very first
 * paint, server-rendered included, and there is nothing to hydrate.
 *
 * The current theme is read from the DOM at click time rather than held in
 * state, because `resolvedTheme` is undefined on the first render and the class
 * on `<html>` is the authoritative value anyway.
 */
export function ThemeToggle() {
  const { setTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle between light and dark theme"
      onClick={() => {
        const isDark = document.documentElement.classList.contains("dark");
        setTheme(isDark ? "light" : "dark");
      }}
      className="size-8 text-muted-foreground hover:text-foreground"
    >
      <Sun className="hidden size-4 dark:block" aria-hidden />
      <Moon className="size-4 dark:hidden" aria-hidden />
    </Button>
  );
}
