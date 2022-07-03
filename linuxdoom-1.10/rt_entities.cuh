#ifndef RT_ENTITIES_CUH_
#define RT_ENTITIES_CUH_

#include "p_mobj.h"
#include <vector>

class SceneEntity;

namespace detail {
    extern std::vector<SceneEntity*> scene_entities_to_free;
}

SceneEntity* RT_CreateMapThing(mobjtype_t type, mobj_t *obj);
void RT_DestroySceneEntity(SceneEntity* entity);
void RT_UpdateEntityPosition(mobj_t *obj);

#endif