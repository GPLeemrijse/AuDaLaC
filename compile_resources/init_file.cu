#include "init_file.h"
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <new>
#include <inttypes.h>
#include "ADL.h"

namespace InitFile {



    void StructInfo::print_info(){
        std::cout << this->name << "(";
        for(auto t : this->parameter_types){
            std::cout << t << ", ";
        }
        std::cout << this->nrof_instances << "):" << std::endl;

        for(int i = 0; i < this->nrof_instances; i++){
            for (int p = 0; p < this->parameter_types.size(); p++){
                if (this->parameter_types[p] == ADL::Int){
                    int32_t* par_value_ptr = (int32_t*)this->parameter_data[p];
                    int32_t par_value = par_value_ptr[i];
                    std::cout << std::to_string(par_value) << " ";

                } else if (this->parameter_types[p] == ADL::Nat){
                    uint32_t* par_value_ptr = (uint32_t*)this->parameter_data[p];
                    uint32_t par_value = par_value_ptr[i];
                    std::cout << std::to_string(par_value) << " ";

                } else if (this->parameter_types[p] == ADL::Bool){
                    bool* par_value_ptr = (bool*)this->parameter_data[p];
                    bool par_value = par_value_ptr[i];
                    std::cout << std::to_string(par_value) << " ";

                } else {
                    uint32_t* par_value_ptr = (uint32_t*)this->parameter_data[p];
                    uint32_t par_value = par_value_ptr[i];
                    std::cout << std::to_string(par_value) << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    std::vector<StructInfo> parse(const char* init_file) {
        std::ifstream infile(init_file);

        std::string line;
        std::getline(infile, line);
        
        /* Parse the header */
        uint nrof_structs;
        if (line.rfind("ADL structures ", 0) == 0) {
            std::istringstream iss(line.substr(14, line.length()));
            iss >> nrof_structs;
        } else {
            throw std::invalid_argument("Invalid header prefix.");
        }

        /* Allocate space for structs */
        std::vector<StructInfo> struct_info(nrof_structs);


        /* Parse the structure parameter declarations */
        for (int s = 0; s < nrof_structs; s++){
            std::getline(infile, line);
            std::stringstream iss(line);

            iss >> struct_info[s].name; // Read in name

            std::string par_name;
            while(iss >> par_name){
                struct_info[s].parameter_types.push_back(ADL::parse_type_string(par_name));
            }
        }

        /* Parse the structure instance declarations */
        for (int strct = 0; strct < nrof_structs; strct++){
            StructInfo* s_info = &struct_info[strct];
            std::getline(infile, line);
            
            std::string name;
            std::string middle_part;
            uint nrof_instances;
            std::istringstream iss(line);
            iss >> name >> middle_part >> nrof_instances;

            bool starts_with_name = name.compare(s_info->name) == 0;
            bool middle_part_instances = middle_part.compare("instances") == 0;
            
            // Error if wrong input received
            if (!starts_with_name || !middle_part_instances || nrof_instances < 1) {
                throw std::invalid_argument("Wrong structure instances header.");
            }

            s_info->nrof_instances = nrof_instances;

            /* Allocate space for instances per parameter */
            uint nrof_params = s_info->parameter_types.size();
            s_info->parameter_data.reserve(nrof_params);

            std::vector<size_t> param_sizes(nrof_params);
            for (int p = 0; p < nrof_params; p++){
                size_t param_size = ADL::size_of_type(s_info->parameter_types[p]);
                void* data_ptr = malloc(nrof_instances * param_size);

                if (data_ptr == NULL){
                    throw std::bad_alloc();
                }
                
                s_info->parameter_data.push_back(data_ptr);
                param_sizes.push_back(param_size);
            }

            for (int inst = 0; inst < nrof_instances; inst++){
                std::getline(infile, line);
                std::istringstream iss(line);
                int64_t par_value;
                for (int p = 0; p < nrof_params; p++){
                    iss >> par_value;

                    if (s_info->parameter_types[p] == ADL::Int){
                        int32_t* par_array = (int32_t*)s_info->parameter_data[p];
                        par_array[inst] = par_value;

                    } else if (s_info->parameter_types[p] == ADL::Nat){
                        uint32_t* par_array = (uint32_t*)s_info->parameter_data[p];
                        par_array[inst] = par_value;

                    } else if (s_info->parameter_types[p] == ADL::Bool){
                        bool* par_array = (bool*)s_info->parameter_data[p];
                        par_array[inst] = par_value;

                    } else {
                        uint32_t* par_array = (uint32_t*)s_info->parameter_data[p];
                        par_array[inst] = par_value; // Refs as indices
                    }
                }
            } 
        }

        return struct_info;
    }
}