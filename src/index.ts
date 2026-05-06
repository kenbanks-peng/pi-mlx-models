import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default async function (pi: ExtensionAPI) {
  pi.registerCommand("mlx-models-help", {
    description: "Show pi-mlx-models extension status",
    handler: async (_args, ctx) => {
      ctx.ui.notify("pi-mlx-models loaded ✅", "success");
      ctx.ui.setStatus("pi-mlx-models", "ready");
    },
  });
}
