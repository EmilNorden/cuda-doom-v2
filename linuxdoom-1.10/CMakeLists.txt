add_definitions(-DNORMALUNIX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${doom_SOURCE_DIR}/cmake")

if (UNIX AND NOT APPLE)
    add_definitions(-DLINUX)
endif ()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")

add_library(doom_lib
        am_map.cpp
        am_map.h
        custom_mix_pitch.cpp
        custom_mix_pitch.h
        custom_mix_pitch_callback.h
        d_englsh.h
        d_event.h
        d_french.h
        d_items.cpp
        d_items.h
        d_main.cpp
        d_main.h
        d_net.cpp
        d_net.h
        d_player.h
        d_textur.h
        d_think.h
        d_ticcmd.h
        doomdata.h
        doomdef.cpp
        doomdef.h
        doomstat.cpp
        doomstat.h
        doomtype.h
        dstrings.cpp
        dstrings.h
        f_finale.cpp
        f_finale.h
        f_wipe.cpp
        f_wipe.h
        g_game.cpp
        g_game.h
        hu_lib.cpp
        hu_lib.h
        hu_stuff.cpp
        hu_stuff.h
        i_net.cpp
        i_net.h
        i_sound.cpp
        i_sound.h
        i_system.cpp
        i_system.h
        i_video.cpp
        i_video.h
        info.cpp
        info.h
        m_argv.cpp
        m_argv.h
        m_bbox.cpp
        m_bbox.h
        m_cheat.cpp
        m_cheat.h
        m_fixed.cpp
        m_fixed.h
        m_menu.cpp
        m_menu.h
        m_misc.cpp
        m_misc.h
        m_random.cpp
        m_random.h
        m_swap.cpp
        m_swap.h
        midi_converter.cpp
        midi_converter.h
        mus_parser.cpp
        mus_parser.h
        p_ceilng.cpp
        p_doors.cpp
        p_enemy.cpp
        p_floor.cpp
        p_inter.cpp
        p_inter.h
        p_lights.cpp
        p_local.h
        p_map.cpp
        p_maputl.cpp
        p_mobj.cpp
        p_mobj.h
        p_plats.cpp
        p_pspr.cpp
        p_pspr.h
        p_saveg.cpp
        p_saveg.h
        p_setup.cpp
        p_setup.h
        p_sight.cpp
        p_spec.cpp
        p_spec.h
        p_switch.cpp
        p_telept.cpp
        p_tick.cpp
        p_tick.h
        p_user.cpp
        r_bsp.cpp
        r_bsp.h
        r_data.cpp
        r_data.h
        r_defs.h
        r_draw.cpp
        r_draw.h
        r_local.h
        r_geometry.cpp
        r_geometry.h
        r_main.cpp
        r_main.h
        r_opengl.cpp
        r_opengl.h
        r_plane.cpp
        r_plane.h
        r_segs.cpp
        r_segs.h
        r_sky.cpp
        r_sky.h
        r_state.h
        r_things.cpp
        r_things.h
        rt_dizzy_effect.cu
        rt_dizzy_effect.cuh
        rt_entities.cu
        rt_entities.cuh
        rt_init_options.cuh
        rt_material.cu
        rt_material.cuh
        rt_raytracing_opengl.cu
        rt_raytracing_opengl.cuh
        rt_raytracing_scene.cu
        rt_raytracing_scene.cuh
        rt_raytracing.cu
        rt_raytracing.cuh
        rt_texture.cu
        rt_texture.cuh
        s_sound.cpp
        s_sound.h
        sounds.cpp
        sounds.h
        st_lib.cpp
        st_lib.h
        st_stuff.cpp
        st_stuff.h
        tables.cpp
        tables.h
        v_video.cpp
        v_video.h
        w_wad.cpp
        w_wad.h
        wi_stuff.cpp
        wi_stuff.h
        z_zone.cpp
        z_zone.h opengl/common.h opengl/common.cpp)

add_executable(doom
        i_main.cpp)

target_link_libraries(doom
        doom_lib)

set_target_properties(doom_lib PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(doom PROPERTIES CUDA_ARCHITECTURES "86")
# target_compile_options(doom PUBLIC -m32 -I/usr/local/opt/gcc/include)
# target_link_options(doom PUBLIC -m32 -L/usr/local/opt/gcc/lib/gcc/9)
# target_compile_options(doom PUBLIC -m32 -I/usr/local/opt/llvm/include)
# target_link_options(doom PUBLIC -m32 -L/usr/local/opt/llvm/lib)

add_subdirectory(renderer)
add_subdirectory(wad)
add_subdirectory(opengl)
add_subdirectory(geometry)

find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(FreeImage REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(fmt REQUIRED)

target_link_libraries(doom_lib
        m
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${FREEIMAGE_LIBRARIES}
        ${SDL2_LIBRARIES}
        ${SDL2_MIXER_LIBRARIES}
        fmt::fmt)

target_include_directories(doom_lib SYSTEM PRIVATE
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${FREEIMAGE_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
        ${SDL2_MIXER_INCLUDE_DIRS}
        ${doom_SOURCE_DIR}/external)

set_target_properties(
        doom_lib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(doom_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        -arch=compute_86
        -code=sm_86
        >)

target_compile_options(doom_lib PRIVATE $<$<CONFIG:DEBUG>:
        -g
        -DDOOM_ASSERTIONS
        >)