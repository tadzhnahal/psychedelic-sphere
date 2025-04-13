import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import ctypes
import time

# Вершинный шейдер
vertex_shader = """
#version 330
in vec3 position;
in vec3 normal;
out vec3 fragNormal;
out vec3 fragPosition;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    fragPosition = vec3(model * vec4(position, 1.0));
    fragNormal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(fragPosition, 1.0);
}
"""

# Фрагментный шейдер
fragment_shader = """
#version 330
in vec3 fragNormal;
in vec3 fragPosition;
out vec4 fragColor;

uniform float time;
uniform vec2 mousePos;      // Позиция мыши в координатах от 0 до 1
uniform int mouseClicked;   // Флаг клика мыши
uniform float waveTime;     // Время с момента клика для волны
uniform float waveDuration; // Общая продолжительность волны

void main() {
    vec3 norm = normalize(fragNormal);
    
    // Базовые цвета в RGB
    float r = 0.5 + 0.5 * sin(time + norm.x * 10.0);
    float g = 0.5 + 0.5 * sin(time + norm.y * 10.0 + 2.0);
    float b = 0.5 + 0.5 * sin(time + norm.z * 10.0 + 4.0);
    
    // Свечение
    float glow = dot(norm, vec3(0.0, 0.0, 1.0)) * 0.5 + 0.5;
    vec3 color = vec3(r, g, b) * glow * 1.5;
    
    // При наведении мыши область вокруг будет чуть плавнее
    vec3 screenPos = vec3(gl_FragCoord.xy / vec2(800, 600), 0.0);
    float distToMouse = distance(screenPos.xy, mousePos);
    
    // Чем меньше область воздействия, тем эффект более мягкий
    if (distToMouse < 0.12) {
        // Квадратичное затухание, чтобы эффект был более плавным
        float intensity = pow(1.0 - distToMouse / 0.12, 2.0) * 0.4;
        // Более тонкий цвет для подсветки
        color = mix(color, vec3(1.0, 0.8, 0.5), intensity);
    }
    
    // Эффект волны при клике
    if (mouseClicked == 1 && waveTime > 0.0) {
        // Более медленная скорость распространения
        float waveSpeed = 0.5;
        // Радиус волны
        float waveRadius = waveTime * waveSpeed;
        
        // Кривая затухания волны
        float progress = waveTime / waveDuration;
        
        // Синусоидальная форма, которая отвечает за появление и затухание волны
        float waveFade = sin(progress * 3.14159) * 0.8;
        
        // Чем дальше волна, тем затухание интенсивнее
        float distanceFade = max(0.0, 1.0 - (waveRadius * 0.4));
        
        // Динамическая ширина волны, которая постепенно увеличивается
        float waveWidth = 0.06 + 0.14 * progress;
        
        if (waveFade > 0.0) {
            // Гауссова кривая, которая помогает сделать край волны более размытым
            float distanceFromWave = abs(distToMouse - waveRadius);
            float gaussianFalloff = exp(-distanceFromWave * distanceFromWave / (2.0 * waveWidth * waveWidth));
            
            // Схлопываем все факторы затухания
            float finalIntensity = gaussianFalloff * waveFade * distanceFade * 0.4;
            
            // Миксим цвет волны
            vec3 waveColor = vec3(0.2, 0.6, 0.9);
            color = mix(color, waveColor, finalIntensity);
        }
    }
    
    fragColor = vec4(color, 1.0);
}
"""

# Компиляция шейдеров
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

# Сама программа
def create_shader_program():
    vs = compile_shader(vertex_shader, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_shader, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
    return program

# Сферические координаты
def create_sphere(res=64):
    vertices = []
    normals = []
    for i in range(res + 1):
        lat = np.pi * i / res
        for j in range(res + 1):
            lon = 2 * np.pi * j / res
            x = np.sin(lat) * np.cos(lon)
            y = np.cos(lat)
            z = np.sin(lat) * np.sin(lon)
            vertices.append((x, y, z))
            normals.append((x, y, z))
    indices = []
    for i in range(res):
        for j in range(res):
            p1 = i * (res + 1) + j
            p2 = p1 + res + 1
            indices += [p1, p2, p1 + 1, p2, p2 + 1, p1 + 1]
    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint32)

def main():
    pygame.init()
    display_width, display_height = 800, 600
    pygame.display.set_mode((display_width, display_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("На все твои аргументы у меня есть бебебебебебебе")
    glEnable(GL_DEPTH_TEST)

    program = create_shader_program()
    glUseProgram(program)

    # Сфера
    vertices, normals, indices = create_sphere(64)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + normals.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, normals.nbytes, normals)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    pos_loc = glGetAttribLocation(program, 'position')
    glEnableVertexAttribArray(pos_loc)
    glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    norm_loc = glGetAttribLocation(program, 'normal')
    glEnableVertexAttribArray(norm_loc)
    glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes))

    # Юниформы всякие
    model_loc = glGetUniformLocation(program, 'model')
    view_loc = glGetUniformLocation(program, 'view')
    proj_loc = glGetUniformLocation(program, 'projection')
    time_loc = glGetUniformLocation(program, 'time')
    mouse_pos_loc = glGetUniformLocation(program, 'mousePos')
    mouse_clicked_loc = glGetUniformLocation(program, 'mouseClicked')
    wave_time_loc = glGetUniformLocation(program, 'waveTime')
    wave_duration_loc = glGetUniformLocation(program, 'waveDuration')

    # Управление сферой
    clock = pygame.time.Clock()
    zoom = -4.0
    target_zoom = -4.0
    t_start = time.time()
    
    # Вращение
    rotation_x = 0
    rotation_y = 0
    mouse_pressed = False
    prev_mouse_pos = None
    
    # Эффект волны от клика
    mouse_clicked = 0
    wave_start_time = 0
    wave_duration = 6.0

    while True:
        dt = clock.tick(60) / 1000.0
        t = time.time() - t_start
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = 1
                    wave_start_time = current_time
                
                if event.button == 3:
                    mouse_pressed = True
                    prev_mouse_pos = pygame.mouse.get_pos()
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_pressed = False
                    
            elif event.type == MOUSEMOTION and mouse_pressed:
                current_mouse_pos = pygame.mouse.get_pos()
                dx = current_mouse_pos[0] - prev_mouse_pos[0]
                dy = current_mouse_pos[1] - prev_mouse_pos[1]
                
                rotation_y += dx * 0.005
                rotation_x += dy * 0.005
                
                prev_mouse_pos = current_mouse_pos

        # Управление
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            target_zoom += 0.1
        if keys[K_s]:
            target_zoom -= 0.1
        if keys[K_LEFT]:
            rotation_y -= 0.03
        if keys[K_RIGHT]:
            rotation_y += 0.03
        if keys[K_UP]:
            rotation_x -= 0.03
        if keys[K_DOWN]:
            rotation_x += 0.03
            
        zoom += (target_zoom - zoom) * 0.1

        wave_time = 0.0
        if mouse_clicked == 1:
            wave_elapsed = current_time - wave_start_time
            if wave_elapsed < wave_duration:
                wave_time = wave_elapsed
            else:
                mouse_clicked = 0

        mouse_x, mouse_y = pygame.mouse.get_pos()
        normalized_mouse_pos = (mouse_x / display_width, 1.0 - mouse_y / display_height)  # Инвертируем Y для OpenGL, потому что без этого не работает, гад

        glClearColor(0.0, 0.0, 0.0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(program)
        
        model = np.identity(4, dtype=np.float32)
        
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, math.cos(rotation_x), -math.sin(rotation_x), 0],
            [0, math.sin(rotation_x), math.cos(rotation_x), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        rot_y = np.array([
            [math.cos(rotation_y), 0, math.sin(rotation_y), 0],
            [0, 1, 0, 0],
            [-math.sin(rotation_y), 0, math.cos(rotation_y), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        model = np.dot(model, rot_y)
        model = np.dot(model, rot_x)

        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, zoom],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        proj = gluPerspective(45, display_width/display_height, 0.1, 100.0)
        proj = glGetFloatv(GL_PROJECTION_MATRIX)

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj)
        glUniform1f(time_loc, t)
        glUniform2f(mouse_pos_loc, normalized_mouse_pos[0], normalized_mouse_pos[1])
        glUniform1i(mouse_clicked_loc, mouse_clicked)
        glUniform1f(wave_time_loc, wave_time)
        glUniform1f(wave_duration_loc, wave_duration)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        pygame.display.flip()

# Фигня пожестче фракталов будет, я считаю, а вообще, в планах сделать какой-нибудь плеер,
# Чтобы эта сфера менялась в зависимости от музыки.
# Как плеер на первых сервис паках XP.
if __name__ == "__main__":
    main()
    