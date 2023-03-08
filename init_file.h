#ifndef INIT_FILE_H
#define INIT_FILE_H

#include <string>
#include <vector>

namespace InitFile {

    struct struct_info_t {
        enum ParameterType {
            Int,
            Nat,
            Bool,
            Ref
        };

        std::string name;
        std::vector<ParameterType> parameter_types;
        uint nrof_instances;
        std::vector<void*> parameter_data;

        static ParameterType parse_type_string(std::string s);

        static constexpr size_t size_of_type(ParameterType t);

        void print_info();

    };

    std::vector<struct_info_t> parse(const char* init_file);

}

#endif