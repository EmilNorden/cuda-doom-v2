#include <catch2/catch_test_macros.hpp>
#include "geometry/polygon.h"

using namespace geometry;

TEST_CASE("Polygon winding order", "[polygon]") {
    std::vector<glm::vec2> verts_clockwise{{0, 0}, {10, 0}, {0, -10}};
    std::vector<glm::vec2> verts_counter_clockwise{{100, 100}, {-100, 200}, {-200, 0}};
    Polygon p_clockwise(verts_clockwise);
    Polygon p_counter_clockwise(verts_counter_clockwise);

    SECTION("Constructor sets correct winding order") {
        REQUIRE(p_clockwise.winding() == Winding::Clockwise);
        REQUIRE(p_counter_clockwise.winding() == Winding::CounterClockwise);
    }

    SECTION("Flipping winding when CCW should yield CW winding") {
        p_counter_clockwise.flip_winding();
        REQUIRE(p_counter_clockwise.winding() == Winding::Clockwise);
        for(int i = 0; i < verts_counter_clockwise.size(); ++i) {
            REQUIRE(p_counter_clockwise[i] == verts_counter_clockwise[verts_counter_clockwise.size() - 1 - i]);
        }
    }

    SECTION("Flipping winding when CW should yield CCW winding") {
        p_clockwise.flip_winding();
        REQUIRE(p_clockwise.winding() == Winding::CounterClockwise);
        for(int i = 0; i < verts_clockwise.size(); ++i) {
            REQUIRE(p_clockwise[i] == verts_clockwise[verts_clockwise.size() - 1 - i]);
        }
    }

    SECTION("Asserting winding order changes winding order if needed") {
        p_clockwise.assert_winding(Winding::CounterClockwise);
        REQUIRE(p_clockwise.winding() == Winding::CounterClockwise);
        for(int i = 0; i < verts_clockwise.size(); ++i) {
            REQUIRE(p_clockwise[i] == verts_clockwise[verts_clockwise.size() - 1 - i]);
        }

        p_counter_clockwise.assert_winding(Winding::Clockwise);
        REQUIRE(p_counter_clockwise.winding() == Winding::Clockwise);
        for(int i = 0; i < verts_counter_clockwise.size(); ++i) {
            REQUIRE(p_counter_clockwise[i] == verts_counter_clockwise[verts_counter_clockwise.size() - 1 - i]);
        }
    }

    SECTION("Asserting winding order does nothing when already using given winding order") {
        p_clockwise.assert_winding(Winding::Clockwise);
        REQUIRE(p_clockwise.winding() == Winding::Clockwise);
        for(int i = 0; i < verts_clockwise.size(); ++i) {
            REQUIRE(p_clockwise[i] == verts_clockwise[i]);
        }

        p_counter_clockwise.assert_winding(Winding::CounterClockwise);
        REQUIRE(p_counter_clockwise.winding() == Winding::CounterClockwise);
        for(int i = 0; i < verts_counter_clockwise.size(); ++i) {
            REQUIRE(p_counter_clockwise[i] == verts_counter_clockwise[i]);
        }
    }
}

TEST_CASE("Polygon contains vertex", "[polygon]") {
    Polygon polygon(
            std::vector<glm::vec2>{{200, 200}, {400, 200}, {400, 400}, {200, 400}});

    // Simple case, point in middle
    REQUIRE(polygon.contains_vertex({300, 300}));

    // Simple case, point completely outside
    REQUIRE_FALSE(polygon.contains_vertex({0, 0}));

    // Points on edge should not count
    REQUIRE_FALSE(polygon.contains_vertex({200, 200}));
    REQUIRE_FALSE(polygon.contains_vertex({400, 200}));
    REQUIRE_FALSE(polygon.contains_vertex({200, 400}));
    REQUIRE_FALSE(polygon.contains_vertex({400, 400}));
}

TEST_CASE("Polygon intersection", "[polygon]") {
    Polygon main_polygon(
            std::vector<glm::vec2>{{200, 200}, {400, 200}, {400, 400}, {200, 400}});
    Polygon intersecting_polygon(
            std::vector<glm::vec2>{{300, 300}, {600, 300}, {600, 600}, {300, 600}});
    Polygon non_intersecting_polygon(
            std::vector<glm::vec2>{{0, 0}, {-10, 0}, {-10, 1000}, {0, -1000}});

    REQUIRE(main_polygon.intersects_polygon(intersecting_polygon));
    REQUIRE_FALSE(main_polygon.intersects_polygon(non_intersecting_polygon));
}

TEST_CASE("Polygon combination", "[polygon]") {
    Polygon polygon(std::vector<glm::vec2>{
            {0,   0},
            {100, 0},
            {100, 200},
            {0,   200}
    });

    SECTION("Should combine both polygons") {
        Polygon child(std::vector<glm::vec2>{
                {1, 1},
                {2, 1},
                {2, 2}
        });

        polygon.combine_with(child);

        REQUIRE(polygon.size() == 9); // 4 (parent) + 3 (child) + 2 (extra edge)
        REQUIRE(polygon[0] == glm::vec2{0, 0}); // Begin with closest point in parent
        REQUIRE(polygon[1] == glm::vec2{1, 1}); // Beginning of child polygon
        REQUIRE(polygon[2] == glm::vec2{2, 1});
        REQUIRE(polygon[3] == glm::vec2{2, 2}); // End of child polygon
        REQUIRE(polygon[4] == glm::vec2{1, 1}); // Back start of child polygon
        REQUIRE(polygon[5] == glm::vec2{0, 0}); // Back to "weld point" of parent polygon
        REQUIRE(polygon[6] == glm::vec2{100, 0});
        REQUIRE(polygon[7] == glm::vec2{100, 200});
        REQUIRE(polygon[8] == glm::vec2{0, 200});
    }

    SECTION("Should combine at closest vertex") {
        Polygon child(std::vector<glm::vec2>{
                {50, 10},
                {90, 10},
                {40, 50}
        });

        polygon.combine_with(child);

        REQUIRE(polygon[0] == glm::vec2{0, 0});
        REQUIRE(polygon[1] == glm::vec2{100, 0});
        REQUIRE(polygon[2] == glm::vec2{90, 10});
    }
}

TEST_CASE("Polygon winding order assertion", "[polygon]") {
    Polygon p_clockwise(std::vector<glm::vec2>{{0, 0}, {10, 0}, {0, -10}});
}