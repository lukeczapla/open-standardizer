// src/gpu/tautomer_motif_tags.hpp

#pragma once

#include <vector>
#include <cstdint>

namespace RDKit {
class ROMol;
}

void tagTautomerMotifs(
    const RDKit::ROMol &mol,
    std::vector<uint16_t> &atomFlags,
    std::vector<uint16_t> &bondFlags
);
