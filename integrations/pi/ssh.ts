/** Pi SSH tools for one TileOPs round.
 *
 * Target syntax: root@127.0.0.1:<port>:/workspace/round
 * The identity is read from TILEOPS_SSH_IDENTITY, which worker-env exports.
 */
import { spawn } from "node:child_process";
import type {
  BashOperations,
  EditOperations,
  ExtensionAPI,
  ReadOperations,
  WriteOperations,
} from "@earendil-works/pi-coding-agent";
import {
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
} from "@earendil-works/pi-coding-agent";

type Target = { remote: string; port?: string; remoteCwd: string };

function parseTarget(value: string): Target {
  const match = value.match(/^(.*?)(?::(\d+))?:(\/.*)$/);
  if (!match) return { remote: value, remoteCwd: "/workspace/round" };
  return { remote: match[1], port: match[2], remoteCwd: match[3] };
}

function sshArgs(target: Target): string[] {
  const args = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"];
  if (target.port) args.push("-p", target.port);
  if (process.env.TILEOPS_SSH_IDENTITY) args.push("-i", process.env.TILEOPS_SSH_IDENTITY);
  args.push(target.remote);
  return args;
}

function runSsh(target: Target, command: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const child = spawn("ssh", [...sshArgs(target), command], { stdio: ["ignore", "pipe", "pipe"] });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on("data", (data) => stdout.push(data));
    child.stderr.on("data", (data) => stderr.push(data));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve(Buffer.concat(stdout));
      else reject(new Error(`SSH failed (${code}): ${Buffer.concat(stderr).toString()}`));
    });
  });
}

function remotePath(path: string, localCwd: string, remoteCwd: string): string {
  if (path === localCwd) return remoteCwd;
  if (path.startsWith(`${localCwd}/`)) return `${remoteCwd}/${path.slice(localCwd.length + 1)}`;
  return path;
}

function readOps(target: Target, localCwd: string): ReadOperations {
  return {
    readFile: (path) => runSsh(target, `cat ${JSON.stringify(remotePath(path, localCwd, target.remoteCwd))}`),
    access: (path) => runSsh(target, `test -r ${JSON.stringify(remotePath(path, localCwd, target.remoteCwd))}`).then(() => {}),
    detectImageMimeType: async () => null,
  };
}

function writeOps(target: Target, localCwd: string): WriteOperations {
  return {
    writeFile: async (path, content) => {
      const encoded = Buffer.from(content).toString("base64");
      await runSsh(target, `echo ${JSON.stringify(encoded)} | base64 -d > ${JSON.stringify(remotePath(path, localCwd, target.remoteCwd))}`);
    },
    mkdir: (path) => runSsh(target, `mkdir -p ${JSON.stringify(remotePath(path, localCwd, target.remoteCwd))}`).then(() => {}),
  };
}

function editOps(target: Target, localCwd: string): EditOperations {
  const read = readOps(target, localCwd);
  const write = writeOps(target, localCwd);
  return { readFile: read.readFile, access: read.access, writeFile: write.writeFile };
}

function bashOps(target: Target, localCwd: string): BashOperations {
  return {
    exec: (command, cwd, { onData, signal, timeout }) => new Promise((resolve, reject) => {
      const remoteCwd = remotePath(cwd, localCwd, target.remoteCwd);
      const child = spawn("ssh", [...sshArgs(target), `cd ${JSON.stringify(remoteCwd)} && ${command}`], {
        stdio: ["ignore", "pipe", "pipe"],
      });
      let timedOut = false;
      const timer = timeout ? setTimeout(() => { timedOut = true; child.kill(); }, timeout * 1000) : undefined;
      child.stdout.on("data", onData);
      child.stderr.on("data", onData);
      const abort = () => child.kill();
      signal?.addEventListener("abort", abort, { once: true });
      child.on("error", reject);
      child.on("close", (code) => {
        if (timer) clearTimeout(timer);
        signal?.removeEventListener("abort", abort);
        if (signal?.aborted) reject(new Error("aborted"));
        else if (timedOut) reject(new Error(`timeout:${timeout}`));
        else resolve({ exitCode: code });
      });
    }),
  };
}

export default function (pi: ExtensionAPI) {
  pi.registerFlag("ssh", { description: "round SSH target", type: "string" });
  const localCwd = process.cwd();
  const localRead = createReadTool(localCwd);
  const localWrite = createWriteTool(localCwd);
  const localEdit = createEditTool(localCwd);
  const localBash = createBashTool(localCwd);
  let target: Target | null = null;

  pi.registerTool({ ...localRead, async execute(id, params, signal, onUpdate) {
    return target ? createReadTool(localCwd, { operations: readOps(target, localCwd) }).execute(id, params, signal, onUpdate) : localRead.execute(id, params, signal, onUpdate);
  }});
  pi.registerTool({ ...localWrite, async execute(id, params, signal, onUpdate) {
    return target ? createWriteTool(localCwd, { operations: writeOps(target, localCwd) }).execute(id, params, signal, onUpdate) : localWrite.execute(id, params, signal, onUpdate);
  }});
  pi.registerTool({ ...localEdit, async execute(id, params, signal, onUpdate) {
    return target ? createEditTool(localCwd, { operations: editOps(target, localCwd) }).execute(id, params, signal, onUpdate) : localEdit.execute(id, params, signal, onUpdate);
  }});
  pi.registerTool({ ...localBash, async execute(id, params, signal, onUpdate) {
    return target ? createBashTool(localCwd, { operations: bashOps(target, localCwd) }).execute(id, params, signal, onUpdate) : localBash.execute(id, params, signal, onUpdate);
  }});
  pi.on("session_start", async (_event, ctx) => {
    const value = pi.getFlag("ssh") as string | undefined;
    if (!value) return;
    target = parseTarget(value);
    ctx.ui.setStatus("ssh", ctx.ui.theme.fg("accent", `SSH: ${value}`));
  });
  pi.on("user_bash", () => target ? { operations: bashOps(target, localCwd) } : undefined);
  pi.on("before_agent_start", async (event) => {
    if (!target) return;
    return { systemPrompt: event.systemPrompt.replace(`Current working directory: ${localCwd}`, `Current working directory: ${target.remoteCwd} via SSH`) };
  });
}
