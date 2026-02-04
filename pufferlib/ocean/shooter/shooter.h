#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

/*
Agent can move up and down, and has to learn to shoot a target that starts at a random position
and then moves deterministically.

The agent has:
-Heading (where it is looking)
-Ability to shoot

It observes:
-target x and y

It can choose to:
-change where it's looking (up or down)
-shoot

We'll start with programming hitscan and basic movement in target.
Then add target acceleration and projectile time

New refactor:
Rather than it choosing to change its heading, it will have a heading angular velocity, and it can either accelerate
or decelerate this velocity.

Heading changes by heading_vel each tick, and it observes heading and heading vel
*/
#define PI 3.14159265

const size_t env_steps = 1000;
const float t_vel = 0.001f;
const float t_radius = 0.02;

int NOOP = 0;
int SHOOT = 1;
int LEFT = 2;
int RIGHT = 3;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;

    float n;
} Log;

typedef struct {
    float x;
    float y;
    float vel;
    float r;
} Target;

typedef struct {
    float x;
    float y;
    float heading;
    float heading_vel;
    int ticks_since_reward;
    int hits_since_reset;
    int ticks_since_shot;
} Agent;

typedef struct {
    Log log;
    Target* targets;
    Agent* agents;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int* tick;

    int num_agents; // does this go here? (NOTE: Could get rid of this, since there's one agent per env, and make a lot of the currently-arrays into lists.)

    int height;
    int width;
} Shooter;

// TODO: handle logging in c_step

void init(Shooter* env) {
    // NOTE: env->num_agents is now initialized in my_init (binding.c). Good.
    env->targets = (Target*)calloc(env->num_agents, sizeof(Target));
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->tick = (int*)calloc(env->num_agents, sizeof(int));
}

void compute_observations(Shooter* env) {
    size_t idx = 0;
    for(size_t i = 0; i < env->num_agents; i++) {
        Agent* a = &env->agents[i];
        Target* t = &env->targets[i];
        env->observations[idx++] = a->x;
        env->observations[idx++] = a->y;
        env->observations[idx++] = a->heading;
        env->observations[idx++] = a->heading_vel;
        env->observations[idx++] = t->x;
        env->observations[idx++] = t->y;
        env->observations[idx++] = t->r;
    }
}

void reset_agent_i(Shooter* env, size_t i, int clear_terminal) {
    Agent* a = &env->agents[i];
    a->x = (float)rand()/RAND_MAX;
    a->y = (float)rand()/RAND_MAX;
    a->heading = (float)rand()/RAND_MAX;
    a->heading_vel = 0;
    a->ticks_since_reward = 0;
    a->hits_since_reset = 0;
    a->ticks_since_shot = 9999;

    env->tick[i] = 0;
    env->rewards[i] = 0.0f;
    if(clear_terminal) {
        env->terminals[i] = 0;
    }
}

void reset_agents(Shooter* env) {
    for(size_t i = 0; i < env->num_agents; i++) {
        reset_agent_i(env, i, true);
    }
}

void reset_target_i(Shooter* env, size_t i) { // agent doesn't observe difference frames/vel, so fixed is necessary to be learnable. Can/will change later. Also can change radius later.
    Target* t = &env->targets[i];
    t->x = (float)rand()/RAND_MAX;
    do {
        t->y = (float)rand()/RAND_MAX;
    } while(t->y < 0.1 || t->y > 0.9);
    t->vel = t_vel;
    t->r = t_radius;
}

void reset_targets(Shooter* env) {
    for(size_t i = 0; i < env->num_agents; i++) {
        reset_target_i(env, i);
    }
}

void c_reset(Shooter* env) {
    /*
    Set agent to random spot and heading
    Set target to random spot and heading

    agent needs to observe it's own pos, heading,
    and target pos

    1 float for pos, 2 for its pos, 2 for target pos
    */

    size_t num_spaces = 7;
    memset(env->observations, 0, num_spaces * env->num_agents * sizeof(float));

    reset_agents(env);
    reset_targets(env);
    compute_observations(env);
}

float clip(float min, float max, float val) {
    if(val < min) return min;
    if(val > max) return max;
    return val;
}

void update_target(Shooter* env) {
    for(size_t i = 0; i < env->num_agents; i++) {
        Target *t = &env->targets[i];
        t->y += t->vel;
        if(t->y > 0.9 || t->y < 0.1) t->vel *= -1;
        t->y = clip(0.1, 0.9, t->y);
    }
}


void c_step(Shooter* env) {
    for(size_t i = 0; i < env->num_agents; i++) {
        env->rewards[i] = 0.0f;
        env->tick[i]++;
        env->terminals[i] = 0;

        Agent* agent = &env->agents[i];
        Target* target = &env->targets[i];
        int act = env->actions[i];
        agent->ticks_since_reward++;
        agent->ticks_since_shot++;
        agent->heading = fmodf(agent->heading + agent->heading_vel, 2 * PI);

        if(act == 1) {
            agent->ticks_since_shot = 0;
            /*
            theta = h * 2pi
            d = <cos(theta), sin(theta)>
            v = (x2 - x1, y2 - y1)
            t = <cos(theta) * (x2 - x1), sin(theta) * (y2 - y1)>

            if t > 0 and ||v||^2 - t^2 < r^2, then we hit.
            */
       
            float theta = 2 * PI * agent->heading;

            float direction_x = cosf(theta);
            float direction_y = sinf(theta);

            float vx = target->x - agent->x;
            float vy = target->y - agent->y;

            float t = direction_x * vx + direction_y * vy;
            float mag_v = vx * vx + vy * vy;
            float r_sq = target->r * target->r;

            if(t > 0 && mag_v - t*t < r_sq) {
                env->rewards[i] = 1.0f;

                reset_target_i(env, i);

                agent->hits_since_reset += 1;
                agent->ticks_since_reward = 0;
            }
        } else if(act == 2) {
            agent->heading_vel += 0.1;
        } else if(act == 3) {
            agent->heading_vel -= 0.1;
        }
        if(env->tick[i] == env_steps) { // terminate after env_steps steps
            env->log.episode_return += agent->hits_since_reset;
            env->log.episode_length += env->tick[i];
            env->log.n++;
            env->terminals[i] = 1;
            reset_agent_i(env, i, 0);
            reset_target_i(env, i);
        }
    }
    update_target(env);
    compute_observations(env);
}


void c_render(Shooter* env) {
    const float render_scale = 0.5f;
    const float agent_radius = 20.0f;
    const float target_radius = render_scale * t_radius * env->width;
    const int shot_flash_frames = 1;

    if(!IsWindowReady()) {
        InitWindow((int)(env->width * render_scale), (int)(env->height * render_scale), "PufferLib Shooter");
        SetTargetFPS(60);
    }

    if(IsKeyDown(KEY_ESCAPE)) { // standard
        exit(0);
    }
    Agent* a = &env->agents[0];
    Target* t = &env->targets[0];

    const float w = env->width * render_scale;
    const float h = env->height * render_scale;
    const float ax = a->x * w;
    const float ay = a->y * h;
    const float tx = t->x * w;
    const float ty = t->y * h;

    BeginDrawing();
    ClearBackground((Color){6, 255, 255, 255}); // copied

    float theta = 2.0f * PI * a->heading;
    float hx = cosf(theta);
    float hy = sinf(theta);
    DrawLineEx((Vector2){ax, ay}, (Vector2){ax + hx * 25.0f, ay + hy * 25.0f}, 2.0f, BLACK);
    DrawCircle(ax + hx * 28.0f, ay + hy * 28.0f, 3.0f, BLACK);

    if(a->ticks_since_shot < shot_flash_frames) {
        DrawLineEx((Vector2){ax, ay}, (Vector2){ax + hx * w, ay + hy * h}, 3.0f, RED);
    }
    

    DrawCircle(tx, ty, target_radius, BLUE);
    DrawCircle(ax, ay, agent_radius, YELLOW);

    EndDrawing();
}


void c_close(Shooter* env) {
    free(env->agents);
    free(env->targets);
    free(env->tick);
    if(IsWindowReady()) {
        CloseWindow();
    }
}