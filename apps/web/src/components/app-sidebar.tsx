"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  BadgeCheck,
  ChevronsUpDown,
  CircleHelp,
  Database,
  FileStack,
  FlaskConical,
  Gauge,
  Hexagon,
  Lightbulb,
  LogOut,
  MessageSquare,
  Newspaper,
  Scale,
  Settings,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  Workflow,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
} from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/lib/auth-context";
import { useWorkspaceCounts } from "@/hooks/use-workspace-counts";

type NavItem = {
  title: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  /** Which live count to show, if any. Resolved at render, never hardcoded. */
  badgeKey?: "runs" | "documents" | "datasets" | "claims";
};

const overview: NavItem[] = [
  { title: "Dashboard", href: "/dashboard", icon: Gauge },
  { title: "New analysis", href: "/runs/new", icon: Sparkles },
  { title: "Runs", href: "/runs", icon: Activity, badgeKey: "runs" },
  { title: "Insights", href: "/insights", icon: Lightbulb, badgeKey: "claims" },
  { title: "Chat", href: "/chat", icon: MessageSquare },
];

const agents: NavItem[] = [
  { title: "All agents", href: "/agents", icon: Hexagon },
  { title: "Finance", href: "/agents/finance", icon: TrendingUp },
  { title: "Risk", href: "/agents/risk", icon: ShieldAlert },
  { title: "Market", href: "/agents/market", icon: Scale },
  { title: "News", href: "/agents/news", icon: Newspaper },
  { title: "Workflow", href: "/agents/workflow", icon: Workflow },
  { title: "Critic", href: "/agents/critic", icon: BadgeCheck },
];

const corpus: NavItem[] = [
  { title: "Documents", href: "/documents", icon: FileStack, badgeKey: "documents" },
  { title: "Datasets", href: "/datasets", icon: Database, badgeKey: "datasets" },
  { title: "Benchmarks", href: "/benchmarks", icon: FlaskConical },
];

/** Is `href` the page we are on, or an ancestor of it? */
function isActive(pathname: string, href: string) {
  if (pathname === href) return true;
  // Guard the segment boundary: without it `/agents` would light up for
  // `/agents-archive`, and `/runs` for `/runs-old`.
  return pathname.startsWith(`${href}/`);
}

function NavSection({
  label,
  items,
  pathname,
  counts,
}: {
  label?: string;
  items: NavItem[];
  pathname: string;
  counts: Record<string, number | undefined>;
}) {
  return (
    <SidebarGroup>
      {label ? (
        <SidebarGroupLabel className="text-2xs font-medium tracking-wide text-muted-foreground/70 uppercase">
          {label}
        </SidebarGroupLabel>
      ) : null}
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((item) => {
            const active = isActive(pathname, item.href);
            const badge = item.badgeKey ? counts[item.badgeKey] : undefined;
            return (
              <SidebarMenuItem key={item.href}>
                <SidebarMenuButton
                  asChild
                  isActive={active}
                  tooltip={item.title}
                  className="text-sm"
                >
                  <Link href={item.href}>
                    <item.icon className="size-4 shrink-0" />
                    <span>{item.title}</span>
                  </Link>
                </SidebarMenuButton>
                {/* Zero is rendered as no badge rather than as "0": an empty
                    corpus should read as empty, not as a count of nothing. */}
                {badge ? (
                  <SidebarMenuBadge className="tabular text-2xs">
                    {badge > 99 ? "99+" : badge}
                  </SidebarMenuBadge>
                ) : null}
              </SidebarMenuItem>
            );
          })}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, signOut } = useAuth();
  const counts = useWorkspaceCounts();

  const workspaceLabel = user?.name?.trim() || "Workspace";

  return (
    <Sidebar collapsible="icon" className="border-r">
      <SidebarHeader className="border-b px-3 py-3">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent"
                >
                  <div className="flex aspect-square size-7 items-center justify-center rounded-md bg-foreground text-background">
                    <Hexagon className="size-4" />
                  </div>
                  <div className="grid flex-1 text-left leading-tight">
                    <span className="truncate text-sm font-semibold">
                      {workspaceLabel}
                    </span>
                    <span className="truncate text-2xs text-muted-foreground">
                      {user ? `Workspace · ${user.role}` : "Workspace"}
                    </span>
                  </div>
                  <ChevronsUpDown className="ml-auto size-3.5 text-muted-foreground" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="start"
                className="w-(--radix-dropdown-menu-trigger-width) min-w-56"
              >
                <DropdownMenuLabel className="text-2xs text-muted-foreground uppercase">
                  Signed in
                </DropdownMenuLabel>
                <DropdownMenuItem disabled className="flex-col items-start gap-0">
                  <span className="text-sm">{user?.name}</span>
                  <span className="text-2xs text-muted-foreground">
                    {user?.email}
                  </span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild className="text-sm">
                  <Link href="/settings">Workspace settings</Link>
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="text-sm"
                  onSelect={async () => {
                    await signOut();
                    router.replace("/login");
                  }}
                >
                  <LogOut className="size-3.5" />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <NavSection items={overview} pathname={pathname} counts={counts} />
        <SidebarSeparator />
        <NavSection label="Agents" items={agents} pathname={pathname} counts={counts} />
        <SidebarSeparator />
        <NavSection label="Corpus" items={corpus} pathname={pathname} counts={counts} />
      </SidebarContent>

      <SidebarFooter className="border-t">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={isActive(pathname, "/settings")}
              tooltip="Settings"
              className="text-sm"
            >
              <Link href="/settings">
                <Settings className="size-4" />
                <span>Settings</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={isActive(pathname, "/help")}
              tooltip="Get help"
              className="text-sm"
            >
              <Link href="/help">
                <CircleHelp className="size-4" />
                <span>Get help</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
