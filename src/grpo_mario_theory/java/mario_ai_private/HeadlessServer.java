package grpo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Locale;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioGame;
import engine.core.MarioTimer;
import engine.core.MarioWorld;
import engine.helper.GameStatus;
import engine.helper.MarioActions;
import engine.helper.SpriteType;
import engine.helper.TileFeature;
import engine.core.MarioSprite;

public class HeadlessServer {
    private static final int EMPTY = 0, SOLID = 1, HAZARD = 2, GOAL = 3;
    private MarioWorld world;
    private MarioAgent expert;
    private int elapsed, maxSteps, obsH, obsW;

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        new HeadlessServer().loop();
    }

    private void loop() throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = br.readLine()) != null) {
            try {
                String[] p = line.trim().split("\\s+");
                if (p.length == 0 || p[0].isEmpty()) continue;
                if (p[0].equals("CLOSE")) break;
                if (p[0].equals("RESET")) reset(p);
                else if (p[0].equals("STEP")) step(Integer.parseInt(p[1]));
                else if (p[0].equals("EXPERT")) expert();
                else System.out.println("ERR unknown_command");
            } catch (Throwable t) {
                System.out.println("ERR " + t.getClass().getSimpleName() + ":" + t.getMessage());
            }
            System.out.flush();
        }
    }

    private void reset(String[] p) throws IOException {
        long seed = Long.parseLong(p[1]);
        int maxTimerMs = 0;
        maxSteps = Integer.parseInt(p[3]);
        obsH = Integer.parseInt(p[4]);
        obsW = Integer.parseInt(p[5]);
        String levelSet = p.length > 7 ? p[7] : "notch";
        String level = loadLevel(levelSet, seed);
        world = new MarioWorld(null);
        world.visuals = false;
        world.initializeLevel(level, maxTimerMs);
        world.mario.isLarge = false;
        world.mario.isFire = false;
        elapsed = 0;
        world.update(new boolean[MarioActions.numberOfActions()]);
        expert = new agents.robinBaumgarten.Agent();
        expert.initialize(new MarioForwardModel(world.clone()), new MarioTimer(MarioGame.maxTime));
        emit(0.0);
    }

    private String loadLevel(String levelSet, long seed) throws IOException {
        int idx = 1 + Math.floorMod(mix(seed), 1000);
        return new String(Files.readAllBytes(Paths.get("levels", levelSet, "lvl-" + idx + ".txt")), StandardCharsets.UTF_8);
    }

    private int mix(long x) {
        x ^= x >>> 33;
        x *= 0xff51afd7ed558ccdL;
        x ^= x >>> 33;
        x *= 0xc4ceb9fe1a85ec53L;
        x ^= x >>> 33;
        return (int)x;
    }

    private void step(int action) {
        world.update(toActions(action));
        elapsed++;
        emit(success() ? 1.0 : 0.0);
    }

    private void expert() {
        System.out.println("ACT\t" + fromActions(expert.getActions(new MarioForwardModel(world.clone()), new MarioTimer(MarioGame.maxTime))));
    }

    private boolean[] toActions(int action) {
        boolean[] a = new boolean[MarioActions.numberOfActions()];
        a[MarioActions.RIGHT.getValue()] = action == 1 || action == 2 || action == 3 || action == 4;
        a[MarioActions.JUMP.getValue()] = action == 2 || action == 4 || action == 6;
        a[MarioActions.SPEED.getValue()] = action == 3 || action == 4;
        a[MarioActions.LEFT.getValue()] = action == 5;
        return a;
    }

    private int fromActions(boolean[] a) {
        boolean right = a[MarioActions.RIGHT.getValue()];
        boolean left = a[MarioActions.LEFT.getValue()];
        boolean jump = a[MarioActions.JUMP.getValue()];
        boolean speed = a[MarioActions.SPEED.getValue()];
        if (right && jump && speed) return 4;
        if (right && speed) return 3;
        if (right && jump) return 2;
        if (right) return 1;
        if (left) return 5;
        if (jump) return 6;
        return 0;
    }

    private boolean success() {
        return world.gameStatus == GameStatus.WIN;
    }

    private boolean death() {
        return world.gameStatus == GameStatus.LOSE;
    }

    private boolean done() {
        return success() || death() || world.gameStatus == GameStatus.TIME_OUT || elapsed >= maxSteps;
    }

    private void emit(double reward) {
        StringBuilder sb = new StringBuilder(12000);
        boolean success = success(), death = death(), done = done();
        boolean timeout = done && !success && !death;
        double distance = Math.max(0.0, world.mario.x - world.level.marioTileX * 16.0) / 16.0;
        sb.append("OBS\t").append(done ? 1 : 0).append('\t').append(reward).append('\t');
        sb.append(success ? 1 : 0).append('\t').append(death ? 1 : 0).append('\t').append(timeout ? 1 : 0).append('\t');
        sb.append(distance).append('\t').append(elapsed).append('\t').append(success ? elapsed : -1).append('\t');
        appendObservation(sb);
        System.out.println(sb.toString());
    }

    private void appendObservation(StringBuilder sb) {
        boolean first = true;
        int cx = (int)(world.mario.x / 16);
        int cy = (int)(world.mario.y / 16);
        for (int ch = 0; ch < 4; ch++) {
            for (int iy = 0; iy < obsH; iy++) {
                int ty = cy - obsH / 2 + iy;
                for (int ix = 0; ix < obsW; ix++) {
                    int tx = cx - obsW / 3 + ix;
                    if (!first) sb.append(',');
                    sb.append(classify(tx, ty) == ch ? "1.0" : "0.0");
                    first = false;
                }
            }
        }
        sb.append(',').append(world.mario.xa / 12.0);
        sb.append(',').append(world.mario.ya / 20.0);
        sb.append(',').append(world.mario.onGround ? "1.0" : "0.0");
        sb.append(',').append(Math.min(1.0, elapsed / (double)Math.max(1, maxSteps)));
    }

    private int classify(int x, int y) {
        if (x >= world.level.exitTileX && Math.abs(y - world.level.exitTileY) <= 2) return GOAL;
        if (x < 0 || x >= world.level.tileWidth || y < 0 || y >= world.level.tileHeight) return EMPTY;
        if (world.level.getSpriteType(x, y) != SpriteType.NONE || hasEnemyAt(x, y)) return HAZARD;
        ArrayList<TileFeature> features = TileFeature.getTileType(world.level.getBlock(x, y));
        return (features.contains(TileFeature.BLOCK_ALL) || features.contains(TileFeature.BLOCK_UPPER) || features.contains(TileFeature.BLOCK_LOWER)) ? SOLID : EMPTY;
    }

    private boolean hasEnemyAt(int x, int y) {
        for (MarioSprite sprite : world.getEnemies()) {
            if ((int)(sprite.x / 16) == x && (int)(sprite.y / 16) == y) return true;
        }
        return false;
    }
}
