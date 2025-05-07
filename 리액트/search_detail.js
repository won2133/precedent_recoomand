import { useState, useEffect } from 'react';
import axios from 'axios'
import { useParams } from "react-router";
import Search_box from './search_box.js';
import Menubar from '../elements/menubar';
import Search_title from './search_titlle.js'
import Search_conts from './search_conts.js';
import Paging from '../elements/page.js';
import Header from './header.js';
import LoadingModal from '../elements/LoadingModal';
function Search_detail({apiUrl}) {
  const [searchData, setSearchData] = useState([]);
  const {kind, query, option, page} = useParams();
  useEffect(() => {
    async function fetchData() {
      const result = await axios.get(
        apiUrl+"/search/" + kind+"/" + query
      );
       setSearchData(result.data);
    }
    fetchData();
  }, [])
  return (
    <div>
    <header id="header"><Header/></header>
    <nav id="nav">&nbsp;</nav>
    <div id="section">
      <Search_box opt={option} q={query}/>
      {(typeof searchData.dic === 'undefined') ? ( <LoadingModal/>) : (
        <div id="div" style={{padding: '20px'}}>
            {kind === "article" ? (<Search_title name="민법 내용" num={searchData.dic.total}/>) : (<Search_title name="판례" num={searchData.dic.total}/>)}
            {searchData.dic.total > 0 ?
                (<Search_conts kind={kind} option={option} list={searchData.dic.name_list} word={searchData.dic.word} page={page} total={searchData.dic.total}/>)
               :(<p>검색 결과가 없습니다.</p>)}
        </div>
      )}
    </div>
    <aside id="aside">&nbsp;</aside>
    <footer></footer>
  </div>
  );
};
export default Search_detail;
