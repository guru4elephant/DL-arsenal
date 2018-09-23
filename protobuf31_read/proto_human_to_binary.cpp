#include <fcntl.h>
#include <unistd.h> // close
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "proto/framework.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
using google::protobuf::Descriptor;
using google::protobuf::DescriptorPool;
using google::protobuf::Message;
using google::protobuf::MessageFactory;
using google::protobuf::Arena;
using google::protobuf::Message;
using google::protobuf::io::FileInputStream;
using namespace std;

namespace program_surgery {
    bool read_proto_from_text(const char* filename, Message* proto){
        int fd = open(filename, O_RDONLY);
        if (fd == -1)
            cerr << "File not found: " << filename;
        google::protobuf::io::FileInputStream* input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }
    
    paddle::framework::proto::ProgramDesc convert_raw_to_binary(const char * filename, 
                                                                const char * output_binary_filename) {
        paddle::framework::proto::ProgramDesc program;
        std::string message_name = "ProgramDesc";
        const Descriptor* d =
            DescriptorPool::generated_pool()->FindMessageTypeByName(message_name);
        Message* msg = MessageFactory::generated_factory()->GetPrototype(d)->New();
        read_proto_from_text(filename, msg);
        std::string binary_str;
        program.SerializeToString(&binary_str);
        ofstream fout_bin(output_binary_filename);
        fout_bin.write((char *)binary_str.c_str(), binary_str.size());
        fout_bin.close();
    }
}

int main(int argc, char *argv[])
{
    paddle::framework::proto::ProgramDesc pb = 
        program_surgery::convert_raw_to_binary(argv[1], argv[2]);
    return 0;
}
