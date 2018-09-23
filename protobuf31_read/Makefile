all: proto_human_to_binary.cpp
	protobuf/bin/protoc --proto_path=./proto proto/framework.proto --cpp_out=proto
	g++ -o proto_human_to_binary proto_human_to_binary.cpp proto/framework.pb.cc -Iproto -Iprotobuf/include -Lprotobuf/lib/ -lprotobuf -lprotobuf-lite -lprotoc

clean:
	rm *~
