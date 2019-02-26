import os
import sys

__all__ = ['MultiSlotDataGenerator']

class DataGenerator(object):
    def __init__(self):
        self._proto_info = None

    def _set_filelist(self, filelist):
        if not isinstance(filelist, list) and not isinstance(filelist, tuple):
            raise ValueError("filelist%s must be in list or tuple type" % type(filelist))
        if not filelist:
            raise ValueError("filelist can not be empty")
        self._filelist = filelist

    def _set_process_num(self, process_num):
        if not isinstance(process_num, int):
            raise ValueError("process_num%s must be in int type" % type(process_num))
        if process_num < 1:
            raise ValueError("process_num can not less than 1")
        self._process_num = process_num

    def _set_line_limit(self, line_limit):
        if not isinstance(line_limit, int):
            raise ValueError("line_limit%s must be in int type" % type(line_limit))
        if line_limit < 1:
            raise ValueError("line_limit can not less than 1")
        self._line_limit = line_limit

    def _set_output_dir(self, output_dir):
        if not isinstance(output_dir, str):
            raise ValueError("output_dir%s must be in str type" % type(output_dir))
        if not output_dir:
            raise ValueError("output_dir can not be empty")
        self._output_dir = output_dir

    def _set_output_prefix(self, output_prefix):
        if not isinstance(output_prefix, str):
            raise ValueError("output_prefix%s must be in str type" % type(output_prefix))
        self._output_prefix = output_prefix

    def _set_output_fill_digit(self, output_fill_digit):
        if not isinstance(output_fill_digit, int):
            raise ValueError("output_fill_digit%s must be in int type" % type(output_fill_digit))
        if output_fill_digit < 1:
            raise ValueError("output_fill_digit can not less than 1")
        self._output_fill_digit = output_fill_digit

    def _set_proto_filename(self, proto_filename):
        if not isinstance(proto_filename, str):
           raise ValueError("proto_filename%s must be in str type" % type(proto_filename))
        if not proto_filename:
            raise ValueError("proto_filename can not be empty")
        self._proto_filename = proto_filename

    def _print_info(self):
        '''
        Print the configuration information
        (Called only in the run_from_stdin function).
        '''
        sys.stderr.write("=" * 16 + " config " + "=" * 16 + "\n")
        sys.stderr.write(" filelist size: %d\n" % len(self._filelist))
        sys.stderr.write(" process num: %d\n" % self._process_num)
        sys.stderr.write(" line limit: %d\n" % self._line_limit)
        sys.stderr.write(" output dir: %s\n" % self._output_dir)
        sys.stderr.write(" output prefix: %s\n" % self._output_prefix)
        sys.stderr.write(" output fill digit: %d\n" % self._output_fill_digit)
        sys.stderr.write(" proto filename: %s\n" % self._proto_filename)
        sys.stderr.write("==== This may take a few minutes... ====\n")

    def _get_output_filename(self, output_index, lock=None):
        '''
        This function is used to get the name of the output file and
        update output_index.

        Args:
            output_index(manager.Value(i)): the index of output file.
            lock(manager.Lock): The lock for processes safe.

        Return:
            Return the name(string) of output file.
        '''
        if lock is not None: lock.acquire()
        file_index = output_index.value
        output_index.value += 1
        if lock is not None: lock.release()
        filename = os.path.join(self._output_dir, self._output_prefix) \
                + str(file_index).zfill(self._output_fill_digit)
        sys.stderr.write("[%d] write data to file: %s\n" % (os.getpid(), filename))
        return filename

    def run_from_stdin(self,
                       is_local=True,
                       hadoop_host=None,
                       hadoop_ugi=None,
                       proto_path=None,
                       proto_filename="data_feed.proto"):
        '''
        This function reads the data row from stdin, parses it with the
        process function, and further parses the return value of the 
        process function with the _gen_str function. The parsed data will
        be wrote to stdout and the corresponding protofile will be
        generated. If local is set to False, the protofile will be
        uploaded to hadoop.

        Args:
            is_local(bool): Whether to execute locally. If it is False, the
                            protofile will be uploaded to hadoop. The
                            default value is True.
            hadoop_host(str): The host name of the hadoop. It should be
                              in this format: "hdfs://${HOST}:${PORT}".
            hadoop_ugi(str): The ugi of the hadoop. It should be in this
                             format: "${USERNAME},${PASSWORD}".
            proto_path(str): The hadoop path you want to upload the
                             protofile to.
            proto_filename(str): The name of protofile. The default value
                                 is "data_feed.proto". It is not
                                 recommended to modify it.
        '''
        if is_local:
            pass
        else:
            if hadoop_ugi is None or \
               hadoop_host is None or \
               proto_path is None:
                 raise ValueError("pls set hadoop_ugi, hadoop_host, and proto_path")
        self._set_proto_filename(proto_filename)
        for line in sys.stdin:
            line_iter = self.process(line)
            for user_parsed_line in line_iter():
                if user_parsed_line == None:
                    continue
                sys.stdout.write(self._gen_str(user_parsed_line))
        if self._proto_info is not None:
            # maybe some task do not catch files
            with open(self._proto_filename, "w") as f:
                f.write(self._get_proto_desc(self._proto_info))
            if is_local == False:
                cmd = "$HADOOP_HOME/bin/hadoop fs" \
                    + " -Dhadoop.job.ugi=" + hadoop_ugi \
                    + " -Dfs.default.name=" + hadoop_host \
                    + " -put " + self._proto_filename + " " + proto_path
                os.system(cmd)

    def _gen_str(self, line):
        '''
        Further processing the output of the process() function rewritten by
        user, outputting data that can be directly read by the datafeed,and
        updating proto_info infomation.

        Args:
            line(str): the output of the process() function rewritten by user.

        Returns:
            Return a string data that can be read directly by the datafeed.
        '''
        raise NotImplementedError("pls use MultiSlotDataGenerator or PairWiseDataGenerator")

    def _combine_infos(self, infos):
        '''
        This function is used to merge proto_info information from different
        processes. In general, the proto_info of each process is consistent.

        Args:
            infos(list): the list of proto_infos from different processes.

        Returns:
            Return a unified proto_info.
        '''
        raise NotImplementedError("pls use MultiSlotDataGenerator or PairWiseDataGenerator")

    def _get_proto_desc(self, proto_info):
        '''
        This function outputs the string of the proto file(can be directly
        written to the file) according to the proto_info information.

        Args:
            proto_info: The proto information used to generate the proto
                        string. The type of the variable will be determined
                        by the subclass. In the MultiSlotDataGenerator,
                        proto_info variable is a list of tuple.

        Returns:
            Returns a string of the proto file.
        '''
        raise NotImplementedError("pls use MultiSlotDataGenerator or PairWiseDataGenerator")

    def process(self, line):
        '''
        This function needs to be overridden by the user to process the 
        original data row into a list or tuple.

        Args:
            line(str): the original data row

        Returns:
            Returns the data processed by the user.
              The data format is list or tuple: 
            [(name, [feasign, ...]), ...] 
              or ((name, [feasign, ...]), ...)
             
            For example:
            [("words", [1926, 08, 17]), ("label", [1])]
              or (("words", [1926, 08, 17]), ("label", [1]))

        Note:
            The type of feasigns must be in int or float. Once the float
            element appears in the feasign, the type of that slot will be
            processed into a float.
        '''
        raise NotImplementedError("pls rewrite this function to return a list or tuple: "
                + "[(name, [feasign, ...]), ...] or ((name, [feasign, ...]), ...)")


class MultiSlotDataGenerator(DataGenerator):
    def _combine_infos(self, infos):
        '''
        This function is used to merge proto_info information from different
        processes. In general, the proto_info of each process is consistent.
        The type of input infos is list, and the type of element of infos is
        tuple. The format of element of infos will be (name, type).

        Args:
            infos(list): the list of proto_infos from different processes.

        Returns:
            Return a unified proto_info.

        Note:
            This function is only called by the run_from_files function, so
            when using the run_from_stdin function(usually used for hadoop),
            the output of the process function(rewritten by the user) does
            not allow that the same field to have both float and int type
            values.
        '''
        proto_info = infos[0]
        for info in infos:
            for index, slot in enumerate(info):
                name, type = slot
                if name != proto_info[index][0]:
                    raise ValueError("combine infos error, pls contact the maintainer of this code~")
                if type == "float" and proto_info[index][1] == "uint64":
                    proto_info[index] = (name, type)
        return proto_info

    def _get_proto_desc(self, proto_info):
        '''
        Generate a string of proto file based on the proto_info information.
        
        The proto_info will be a list of tuples:
            >>> [(Name, Type), ...]
        
        The string of proto file will be in this format:
            >>> name: "MultiSlotDataFeed"
            >>> batch_size: 32
            >>> multi_slot_desc {
            >>>     slots {
            >>>         name: Name
            >>>         type: Type
            >>>         is_dense: false
            >>>         is_used: false
            >>>     }
            >>> }

        Args:
            proto_info(list): The proto information used to generate the
                              proto string.

        Returns:
            Returns a string of the proto file.
        '''
        proto_str = "name: \"MultiSlotDataFeed\"\n" \
                + "batch_size: 32\nmulti_slot_desc {\n"
        for elem in proto_info:
            proto_str += "  slots {\n" \
                       + "    name: \"%s\"\n" % elem[0]\
                       + "    type: \"%s\"\n" % elem[1]\
                       + "    is_dense: false\n" \
                       + "    is_used: false\n" \
                       + "  }\n"
        proto_str += "}"
        return proto_str

    def _gen_str(self, line):
        '''
        Further processing the output of the process() function rewritten by
        user, outputting data that can be directly read by the MultiSlotDataFeed,
        and updating proto_info infomation.

        The input line will be in this format:
            >>> [(name, [feasign, ...]), ...] 
            >>> or ((name, [feasign, ...]), ...)
        The output will be in this format:
            >>> [ids_num id1 id2 ...] ...
        The proto_info will be in this format:
            >>> [(name, type), ...]
        
        For example, if the input is like this:
            >>> [("words", [1926, 08, 17]), ("label", [1])]
            >>> or (("words", [1926, 08, 17]), ("label", [1]))
        the output will be:
            >>> 3 1234 2345 3456 1 1
        the proto_info will be:
            >>> [("words", "uint64"), ("label", "uint64")]

        Args:
            line(str): the output of the process() function rewritten by user.

        Returns:
            Return a string data that can be read directly by the MultiSlotDataFeed.
        '''
        if not isinstance(line, list) and not isinstance(line, tuple):
            raise ValueError("the output of process() must be in list or tuple type")
        output = ""
    
        if self._proto_info is None:
            self._proto_info = []
            for item in line:
                name, elements = item
                if not isinstance(name, str):
                    raise ValueError("name%s must be in str type" % type(name))
                if not isinstance(elements, list):
                    raise ValueError("elements%s must be in list type" % type(elements))
                if not elements:
                    raise ValueError("the elements of each field can not be empty, you need padding it in process().")
                self._proto_info.append((name, "uint64"))
                if output: 
                    output += " "
                output += str(len(elements))
                for elem in elements:
                    if isinstance(elem, float):
                        self._proto_info[-1] = (name, "float")
                    elif not isinstance(elem, int) and not isinstance(elem, long):
                        raise ValueError("the type of element%s must be in int or float" % type(elem))
                    output += " " + str(elem)
        else:
            if len(line) != len(self._proto_info):
                raise ValueError("the complete field set of two given line are inconsistent.")
            for index, item in enumerate(line):
                name, elements = item
                if not isinstance(name, str):
                    raise ValueError("name%s must be in str type" % type(name))
                if not isinstance(elements, list):
                    raise ValueError("elements%s must be in list type" % type(elements))
                if not elements:
                    raise ValueError("the elements of each field can not be empty, you need padding it in process().")
                if name != self._proto_info[index][0]:
                    raise ValueError("the field name of two given line are not match: require<%s>, get<%d>."
                            % (self._proto_info[index][0], name))
                if output:
                    output += " "
                output += str(len(elements))
                for elem in elements:
                    if self._proto_info[index][1] != "float":
                        if isinstance(elem, float):
                            self._proto_info[index] = (name, "float")
                        elif not isinstance(elem, int) and not isinstance(elem, long):
                            raise ValueError("the type of element%s must be in int or float" % type(elem))
                    output += " " + str(elem)
        return output + "\n"
