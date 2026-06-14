#include "mds_proto.h"
#include <gtest/gtest.h>
using namespace dfkv;  // NOLINT

TEST(MdsProto, MemberReqRoundTrip) {
  MemberInfo m{"n1", "10.0.0.7", 28000, 5};
  std::string buf = EncodeMemberReq("group-a", m);
  std::string group; MemberInfo got;
  ASSERT_TRUE(DecodeMemberReq(buf.data(), buf.size(), &group, &got));
  EXPECT_EQ(group, "group-a");
  EXPECT_EQ(got, m);
}

TEST(MdsProto, RejectsTruncated) {
  MemberInfo m{"n1", "10.0.0.7", 28000, 5};
  std::string buf = EncodeMemberReq("g", m);
  std::string group; MemberInfo got;
  for (size_t cut = 0; cut < buf.size(); ++cut)
    EXPECT_FALSE(DecodeMemberReq(buf.data(), cut, &group, &got)) << "cut=" << cut;
  EXPECT_TRUE(DecodeMemberReq(buf.data(), buf.size(), &group, &got));
}
