#ifndef INIT_FILE_H
#define INIT_FILE_H

#include <string>
#include <vector>
#include "ADL.h"

namespace InitFile {

    class StructInfo {
        public:

        std::string name;
        std::vector<ADL::Type> parameter_types;
        ADL::inst_size nrof_instances;
        std::vector<void*> parameter_data;

        void print_info();

    };

    std::vector<StructInfo> parse(const char* init_file);
}

#endif