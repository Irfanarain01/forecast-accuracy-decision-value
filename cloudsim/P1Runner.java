import org.cloudsimplus.allocationpolicies.*; import org.cloudsimplus.brokers.*; import org.cloudsimplus.cloudlets.*; import org.cloudsimplus.core.*; import org.cloudsimplus.datacenters.*;
import org.cloudsimplus.hosts.*; import org.cloudsimplus.resources.*; import org.cloudsimplus.schedulers.cloudlet.*; import org.cloudsimplus.utilizationmodels.*; import org.cloudsimplus.vms.*;
import java.io.*; import java.nio.file.*; import java.util.*;

/** Paper 1 power-management episodes executed in CloudSim Plus. Host counts per slot come from kseq.csv (computed by p1_inputs.py). */
public class P1Runner {
    static final double[][] TYPES = {{10000, 60, 150}, {20000, 90, 250}, {40000, 150, 450}}; static final double SLOT = 10.0, SLEEP = 0.05, T0 = 10.0;
    static int NH, VPH; static double W; static double[] HM, PI, PM; static int[] RANK; static double meanVm;

    static double[] runEpisode(int[] kseq, int nSlots, double[][] tk /* slot, arrival, length, deadline */) throws Exception {
        int n = tk.length; boolean[] on = new boolean[NH], drain = new boolean[NH]; Arrays.fill(on, true); double[] ready = new double[NH], hostEnd = new double[NH], since = new double[NH], awake = new double[NH];
        double[][] vmFree = new double[NH][VPH]; int wakes = 0; double[] lastNow = {0};
        CloudSimPlus sim = new CloudSimPlus(0.01); sim.terminateAt(T0 + nSlots * SLOT + 200000);
        List<Host> hosts = new ArrayList<>(); for (int h = 0; h < NH; h++) { List<Pe> pes = new ArrayList<>(); for (int j = 0; j < VPH; j++) pes.add(new PeSimple(HM[h] / VPH)); hosts.add(new HostSimple(65536, 100000, 1000000, pes)); }
        final Map<Vm, Host> vmHost = new IdentityHashMap<>(); VmAllocationPolicySimple alloc = new VmAllocationPolicySimple(); alloc.setFindHostForVmFunction((p, v) -> Optional.ofNullable(vmHost.get(v)));
        DatacenterSimple dc = new DatacenterSimple(sim, hosts, alloc); dc.setSchedulingInterval(SLOT);
        DatacenterBrokerSimple br = new DatacenterBrokerSimple(sim); br.setShutdownWhenIdle(false); br.setVmDestructionDelay(1e9);
        Vm[][] V = new Vm[NH][VPH]; List<Vm> vms = new ArrayList<>();
        for (int h = 0; h < NH; h++) for (int j = 0; j < VPH; j++) { Vm vm = new VmSimple(HM[h] / VPH, 1); vm.setRam(256).setBw(100).setSize(1000); vm.setCloudletScheduler(new CloudletSchedulerSpaceShared()); V[h][j] = vm; vms.add(vm); vmHost.put(vm, hosts.get(h)); }
        br.submitVmList(vms);
        Cloudlet[] cl = new Cloudlet[n]; int[] hostOf = new int[n]; int[] next = {0}, slot = {0}; int[] wk = {0}; boolean[] done = {false};
        sim.addOnClockTickListener(ev -> {
            while (slot[0] < nSlots && sim.clock() >= T0 + slot[0] * SLOT - 1e-9) {
                int i = slot[0]; int k = kseq[i];
                while (next[0] < n && (int) tk[next[0]][0] == i) {
                    int id = next[0]++; double a = tk[id][1], len = tk[id][2]; lastNow[0] = a;
                    // advance: drained hosts that finished sleep at their end time
                    for (int h = 0; h < NH; h++) if (on[h] && drain[h] && hostEnd[h] <= a) { awake[h] += hostEnd[h] - since[h]; on[h] = false; drain[h] = false; }
                    // apply target (once per task arrival, like the reference simulator)
                    Set<Integer> wanted = new HashSet<>(); for (int r = 0; r < k; r++) wanted.add(RANK[r]);
                    for (int h = 0; h < NH; h++) { if (wanted.contains(h)) { if (!on[h]) { on[h] = true; since[h] = (Math.abs(a - Math.floor(a / SLOT) * SLOT) < 1e9 ? Math.floor(a / SLOT) * SLOT : a); ready[h] = since[h] + W; wk[0]++; } else drain[h] = false; }
                        else if (on[h]) { if (hostEnd[h] <= a) { awake[h] += a - since[h]; on[h] = false; drain[h] = false; } else drain[h] = true; } }
                    double bestF = Double.MAX_VALUE; int bh = -1, bj = -1; double bs = 0;
                    for (int h = 0; h < NH; h++) { double hr = on[h] ? ready[h] : a + W; for (int j = 0; j < VPH; j++) { double st = Math.max(Math.max(a, vmFree[h][j]), hr); double f = st + len / (HM[h] / VPH); if (f < bestF - 1e-9) { bestF = f; bh = h; bj = j; bs = st; } } }
                    if (!on[bh]) { on[bh] = true; drain[bh] = false; since[bh] = a; ready[bh] = a + W; wk[0]++; }
                    vmFree[bh][bj] = bestF; hostEnd[bh] = Math.max(hostEnd[bh], bestF); hostOf[id] = bh;
                    Cloudlet c = new CloudletSimple(Math.max(1L, (long) Math.ceil(len)), 1); c.setUtilizationModelCpu(new UtilizationModelFull()); c.setVm(V[bh][bj]);
                    c.setSubmissionDelay(Math.max(0.0, Math.max(a + T0, ready[bh] + T0) - sim.clock())); cl[id] = c; br.submitCloudlet(c);
                }
                slot[0]++;
            }
            if (!done[0] && slot[0] >= nSlots && next[0] >= n) { boolean all = true; for (int id = 0; id < n; id++) if (cl[id] != null && !cl[id].isFinished()) { all = false; break; } if (all) { done[0] = true; sim.terminate(); } }
        });
        sim.start();
        double mk = 0, dyn = 0, resp = 0, wait = 0, execSum = 0; int late = 0;
        for (int id = 0; id < n; id++) { Cloudlet c = cl[id]; double fin = c.getFinishTime() - T0; double a = tk[id][1]; mk = Math.max(mk, fin); resp += fin - a; wait += (c.getStartTime() - T0) - a;
            double ex = c.getTotalExecutionTime(); execSum += ex; int h = hostOf[id]; dyn += (PM[h] - PI[h]) / VPH * ex; if (!c.isFinished() || fin > tk[id][3]) late++; }
        double Tend = Math.max(mk, lastNow[0]); double idle = 0;
        for (int h = 0; h < NH; h++) { double aw = awake[h]; if (on[h]) { double end = Math.max(lastNow[0], hostEnd[h]); aw += end - since[h]; } aw = Math.min(aw, Tend); idle += PI[h] * aw + SLEEP * PI[h] * (Tend - aw); }
        return new double[]{n, mk, idle, dyn, late, resp, wait, execSum, wk[0]};
    }

    public static void main(String[] a) throws Exception {
        NH = Integer.parseInt(a[0]); VPH = Integer.parseInt(a[1]); W = Double.parseDouble(a[2]); String dir = a[3]; String outF = a[4]; String filter = a.length > 5 ? a[5] : "";
        HM = new double[NH]; PI = new double[NH]; PM = new double[NH]; double tot = 0; for (int h = 0; h < NH; h++) { HM[h] = TYPES[h % 3][0]; PI[h] = TYPES[h % 3][1]; PM[h] = TYPES[h % 3][2]; tot += HM[h]; } meanVm = tot / (NH * VPH);
        Integer[] idx = new Integer[NH]; for (int i = 0; i < NH; i++) idx[i] = i; Arrays.sort(idx, (x, y) -> Double.compare(HM[y] / PM[y], HM[x] / PM[x])); RANK = new int[NH]; for (int i = 0; i < NH; i++) RANK[i] = idx[i];
        // episodes
        List<String> el = Files.readAllLines(Paths.get(dir + "/episodes.csv")); el.remove(0); List<String[]> eps = new ArrayList<>(); for (String l : el) eps.add(l.split(","));
        Map<String, List<double[]>> T = new HashMap<>(); List<String> tl = Files.readAllLines(Paths.get(dir + "/tasks.csv")); tl.remove(0);
        for (String l : tl) { String[] p = l.split(","); T.computeIfAbsent(p[0] + "|" + p[1], x -> new ArrayList<>()).add(new double[]{Double.parseDouble(p[2]), Double.parseDouble(p[3]), Double.parseDouble(p[4]), Double.parseDouble(p[5])}); }
        PrintWriter pw = new PrintWriter(new FileWriter(outF)); pw.println("config,part,ep,N,makespan,idle_j,dyn_j,late,resp,wait,exec,wakes");
        try (BufferedReader br = new BufferedReader(new FileReader(dir + "/kseq.csv"))) { br.readLine(); String line;
            while ((line = br.readLine()) != null) { int c1 = line.indexOf(','), c2 = line.indexOf(',', c1 + 1); String cfg = line.substring(0, c1), part = line.substring(c1 + 1, c2); if (!filter.isEmpty() && !cfg.matches(filter)) continue;
                String[] ks = line.substring(c2 + 1).split(" "); int[] kseq = new int[ks.length]; for (int i = 0; i < ks.length; i++) kseq[i] = Integer.parseInt(ks[i]);
                for (String[] e : eps) { if (!e[0].equals(part)) continue; int ep = Integer.parseInt(e[1]), st = Integer.parseInt(e[2]); List<double[]> L = T.get(part + "|" + ep); if (L == null) continue;
                    double[][] tk = L.toArray(new double[0][]); int[] sub = Arrays.copyOfRange(kseq, st, st + 100);
                    double[] r = runEpisode(sub, 100, tk); pw.printf("%s,%s,%d,%.0f,%.3f,%.3f,%.3f,%.0f,%.3f,%.3f,%.3f,%.0f%n", cfg, part, ep, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]); } pw.flush(); } }
        pw.close();
    }
}
