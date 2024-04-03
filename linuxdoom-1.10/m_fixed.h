// Emacs style mode select   -*- C++ -*- 
//-----------------------------------------------------------------------------
//
// $Id:$
//
// Copyright (C) 1993-1996 by id Software, Inc.
//
// This source is available for distribution and/or modification
// only under the terms of the DOOM Source Code License as
// published by id Software. All rights reserved.
//
// The source is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// FITNESS FOR A PARTICULAR PURPOSE. See the DOOM Source Code License
// for more details.
//
// DESCRIPTION:
//	Fixed point arithemtics, implementation.
//
//-----------------------------------------------------------------------------


#ifndef __M_FIXED__
#define __M_FIXED__

#include <limits>

#ifdef __GNUG__
#pragma interface
#endif


//
// Fixed point, 32bit as 16.16.
//
#define FRACBITS		16
#define FRACUNIT		(1<<FRACBITS)

typedef int fixed_t;

fixed_t FixedMul	(fixed_t a, fixed_t b);
fixed_t FixedDiv	(fixed_t a, fixed_t b);
fixed_t FixedDiv2	(fixed_t a, fixed_t b);

template <typename T>
fixed_t FixedFromInteger(T value) {
    bool is_negative = value < 0;
    fixed_t result = abs(value) << FRACBITS;

    // Special case for extreme value
    if(value == std::numeric_limits<short>::min()) {
        // 32768 left shifted by 16 would flip the sign bit and give us a negative value already.
        // Trying to negate this would be UB.
        return result;
    }

    if(is_negative) {
        result = -result;
    }

    return result;
}



#endif
//-----------------------------------------------------------------------------
//
// $Log:$
//
//-----------------------------------------------------------------------------
