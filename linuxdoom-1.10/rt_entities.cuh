#ifndef RT_ENTITIES_CUH_
#define RT_ENTITIES_CUH_

#include "p_mobj.h"

class SceneEntity;
SceneEntity* RT_CreateMapThing(mobjtype_t type, mobj_t *obj);
void RT_DestroySceneEntity(SceneEntity* entity);
void RT_UpdateEntityPosition(mobj_t *obj);

#endif